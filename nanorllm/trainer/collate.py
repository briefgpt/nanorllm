from typing import Any
from nanorllm.utils.util import render_prompt_messages, format_training_text
import torch



# def tokenize_sample(
#     sample: dict[str, Any],
#     tokenizer,
#     max_length: int,
# ) -> dict[str, torch.Tensor]:
#     '''
#     把一条样本变成单条可训练序列，同时保留prompt和response的分界线，用于构建reponse_mask。
#     输入sample:
#     {
#         "prompt_messages": [...],
#         "response": "...",
#         "advantage": 0.5,
#     }
#     1. 分别tokenize prompt和response，获取input_ids，
#     2. 将prompt和response的input_ids 拼接，在序列末尾增加eos_token
#     3. 使用max_length 截断，确定prompt_ids 的长度
#     4. 构建attention_mask
#     5. 构建labels
#     '''
#     prompt_text, response_text = format_training_text(sample)
    
#     prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_attention_mask=False)['input_ids']
#     response_ids = tokenizer(response_text, add_special_tokens=False, return_attention_mask=False)['input_ids']
#     input_id_list = prompt_ids+ response_ids
#     if tokenizer.eos_token_id is not None:
#         input_id_list.append(tokenizer.eos_token_id) # 注意这里手动加了eos token
#     input_id_list = input_id_list[:max_length]
#     prompt_ids_len = min(len(prompt_ids), len(input_id_list))

#     input_ids = torch.tensor(input_id_list, dtype=torch.long)
#     attention_mask = torch.ones_like(input_ids)
#     labels = input_ids.clone()
#     rollout_logprobs = sample['rollout_logprobs']
#     print(sample['advantage'])
#     return {'input_ids': input_ids, 'prompt_input_ids_len': prompt_ids_len, 'labels': labels, 'attention_mask': attention_mask, 
#             'advantage': torch.tensor(sample['advantage'], dtype=torch.float32),
#             'rollout_logprobs': rollout_logprobs}


def build_response_mask(
    input_ids: torch.Tensor,
    prompt_input_ids_len: int,
) -> torch.Tensor:
    '''
    根据input_ids 和 prompt_ids 长度，构造response_mask
    response_mask 和 attention_mask 的区别：
    attention_mask：参与模型前向的token（self.model(input_ids=input_ids, attention_mask=attention_mask)）
    response_mask：参与 policy loss的token（seq_probs = masked_sequence_logprobs(token_probs, batch['response_mask'])）
    '''
    response_mask = torch.zeros_like(input_ids) 
    response_start = min(prompt_input_ids_len, input_ids.shape[0])
    response_mask[response_start:] = 1.0
    
    return response_mask




def collate_train_batch(
    samples: list[dict[str, Any]],
    tokenizer,
    args
) -> dict[str, torch.Tensor]:
    '''
    把step-level的 train sample 整理成模型需要的batch。
    1. 逐条做tokenization，并获取response_mask
    2. 找到这批样本的最长序列，用来后续的padding补齐
    3. 把所有字段pad后stack成batch tensor(用0或者pad_token_id 来补齐)
    4. 最后返回一个字典
    input_ids: [B, T]
    attention_mask: [B, T]
    labels: [B, T]
    response_mask: [B, T]
    advantages: [B]
    '''
    pad_token_id = tokenizer.pad_token_id 

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_response_mask = []
    batch_advantages = []
    batch_rollout_logprobs = []
    
    for sample in samples:
        prompt_ids = sample["prompt_ids"]
        response_ids = sample["response_ids"]
        rollout_logprobs = sample["rollout_logprobs"]

        if prompt_ids.dim() == 2:
            prompt_ids = prompt_ids.squeeze(0)
        if response_ids.dim() == 2:
            response_ids = response_ids.squeeze(0)
        if rollout_logprobs.dim() == 2:
            rollout_logprobs = rollout_logprobs.squeeze(0)

        # Keep rollout logprobs aligned to the response tokens after truncation.
        if response_ids.shape[0] > args.max_length:
            response_ids = response_ids[: args.max_length]
            rollout_logprobs = rollout_logprobs[: args.max_length]
            prompt_ids = prompt_ids[:0]

        total_len = prompt_ids.shape[0] + response_ids.shape[0]
        if total_len > args.max_length:
            keep_prompt_len = max(args.max_length - response_ids.shape[0], 0)
            prompt_ids = prompt_ids[-keep_prompt_len:]

        prompt_input_ids_len = prompt_ids.shape[0]
        input_ids = torch.concat([prompt_ids, response_ids], dim=0)
        attention_mask = torch.ones_like(input_ids)
        response_mask = build_response_mask(input_ids, prompt_input_ids_len)
        labels = input_ids.clone()
        old_logprobs = torch.concat(
            [torch.zeros_like(prompt_ids, dtype=torch.float32), rollout_logprobs],
            dim=0,
        )
        
        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)
        batch_advantages.append(torch.tensor(sample["advantage"], dtype=torch.float32))
        batch_rollout_logprobs.append(old_logprobs)
        batch_response_mask.append(response_mask)

    max_len = max(item.shape[0] for item in batch_input_ids)

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    padded_response_mask = []
    padded_old_logprobs = []

    for input_ids, attention_mask, labels, response_mask, old_logprobs in zip(
        batch_input_ids,
        batch_attention_mask,
        batch_labels,
        batch_response_mask,
        batch_rollout_logprobs,
        strict=True,
    ):
        pad_len = max_len - input_ids.shape[0]
        padded_input_ids.append(torch.nn.functional.pad(input_ids, (0, pad_len), value=pad_token_id))
        padded_attention_mask.append(torch.nn.functional.pad(attention_mask, (0, pad_len), value=0))
        padded_labels.append(torch.nn.functional.pad(labels, (0, pad_len), value=pad_token_id))
        padded_response_mask.append(torch.nn.functional.pad(response_mask, (0, pad_len), value=0))
        padded_old_logprobs.append(torch.nn.functional.pad(old_logprobs, (0, pad_len), value=0.0))

    
    return {
        "input_ids": torch.stack(padded_input_ids, dim=0),
        "attention_mask": torch.stack(padded_attention_mask, dim=0),
        "labels": torch.stack(padded_labels, dim=0),
        "loss_mask": torch.stack(padded_response_mask, dim=0)[:, 1:],
        "response_mask": torch.stack(padded_response_mask, dim=0),
        "advantages": torch.stack(batch_advantages, dim=0) ,
        "old_logprobs": torch.stack(padded_old_logprobs, dim=0)[:, 1:]
    }
