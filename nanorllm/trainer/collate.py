from typing import Any

import torch


def render_prompt_messages(prompt_messages: list[dict[str, Any]]) -> str:
    rendered_messages: list[str] = []

    for message in prompt_messages or []:
        role = str(message.get("role", "user")).strip().upper()
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        rendered_messages.append(f"<{role}>\n{content.strip()}")

    return "\n\n".join(part for part in rendered_messages if part).strip()


def format_training_text(sample: dict[str, Any]) -> dict[str, str]:
    return {
        "prompt_text": render_prompt_messages(sample.get("prompt_messages", [])),
        "response_text": str(sample.get("response", "")),
    }


def tokenize_sample(
    sample: dict[str, Any],
    tokenizer,
    max_length: int,
) -> dict[str, torch.Tensor]:
    formatted = format_training_text(sample)
    prompt_text = formatted["prompt_text"].strip()
    response_text = formatted["response_text"]
    if prompt_text:
        prompt_text = f"{prompt_text}\n\n<Assistant>\n"
    else:
        prompt_text = "<Assistant>\n"


    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_attention_mask=False)['input_ids']
    response_ids = tokenizer(response_text, add_special_tokens=False, return_attention_mask=False)['input_ids']
    input_id_list = prompt_ids+ response_ids
    if tokenizer.eos_token_id is not None:
        input_id_list.append(tokenizer.eos_token_id) # 注意这里手动加了eos token
    input_id_list = input_id_list[:max_length]
    prompt_ids_len = min(len(prompt_ids), len(input_id_list))

    input_ids = torch.tensor(input_id_list, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {'input_ids': input_ids, 'prompt_input_ids_len': prompt_ids_len, 'labels': labels, 'attention_mask': attention_mask, 'advantage': torch.tensor(sample['advantage'], dtype=torch.float32)}


def build_response_mask(
    input_ids: torch.Tensor,
    prompt_input_ids_len: int,
) -> torch.Tensor:

    response_mask = torch.zeros_like(input_ids) 
    response_start = min(prompt_input_ids_len, input_ids.shape[0])
    response_mask[response_start:] = 1.0
    
    return response_mask


def collate_train_batch(
    samples: list[dict[str, Any]],
    tokenizer,
    max_length: int,
) -> dict[str, torch.Tensor]:
    max_len = 0
    tokenized_samples = []


    for sample in samples:
        tokenized_sample = tokenize_sample(sample, tokenizer, max_length)
        tokenized_sample['response_mask'] = build_response_mask(tokenized_sample['input_ids'], tokenized_sample['prompt_input_ids_len'])
        if tokenized_sample['input_ids'].shape[0] > max_len:
            max_len = tokenized_sample['input_ids'].shape[0]
        tokenized_samples.append(tokenized_sample)

    pad_token_id = tokenizer.pad_token_id 

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_response_mask = []
    batch_advantages = []
    
    for item in tokenized_samples:
        pad_len = max_len - item['input_ids'].shape[0]
        input_ids = torch.nn.functional.pad(item['input_ids'], (0, pad_len), value=pad_token_id)
        attention_mask = torch.nn.functional.pad(item['attention_mask'], (0, pad_len), value=0)
        labels = torch.nn.functional.pad(
            item["labels"],
            (0, pad_len),
            value=pad_token_id,
        )
        response_mask = torch.nn.functional.pad(
            item["response_mask"],
            (0, pad_len),
            value=0.0,
        )
        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)
        batch_response_mask.append(response_mask)
        batch_advantages.append(item["advantage"])

    
    return {
        "input_ids": torch.stack(batch_input_ids, dim=0),
        "attention_mask": torch.stack(batch_attention_mask, dim=0),
        "labels": torch.stack(batch_labels, dim=0),
        "response_mask": torch.stack(batch_response_mask, dim=0),
        "advantages": torch.stack(batch_advantages, dim=0),
    }