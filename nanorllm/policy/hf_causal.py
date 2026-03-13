import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanorllm.policy.base import BasePolicy


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model.to(device)


class HFCausalPolicy(BasePolicy):
    def __init__(self, model_name: str, device: str):
        super().__init__(model_name=model_name, device=device)
        self._tokenizer = load_tokenizer(model_name)
        self._model = load_model(model_name, device)

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
    

    def generate(self, prompt_text: str, max_new_tokens: int, temperature: float=1.0, top_p: float | None=None):
        # Todo kvcache enhance
        response_ids = []
        rollout_logprobs = []

        tokenized_prompt = self._tokenizer(prompt_text,add_special_tokens=False, return_tensors="pt")
        input_ids = tokenized_prompt['input_ids']
        attention_mask = tokenized_prompt['attention_mask']
        prompt_ids = tokenized_prompt['input_ids']

        self.model.eval()
        with torch.no_grad():
            for i in range(max_new_tokens):
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[:,-1,:]
                token_logit = torch.nn.functional.softmax(logits/temperature, dim=-1) # 这里是softmax

                token_id = torch.multinomial(token_logit, num_samples=1)

                token_log_logits = torch.nn.functional.log_softmax(logits/temperature, dim=-1)
                token_log_prob = torch.gather(token_log_logits, index=token_id, dim=-1)

                rollout_logprobs.append(token_log_prob)

                response_ids.append(token_id)
                if token_id.item() == self._tokenizer.eos_token_id:
                    break

                input_ids = torch.concat([input_ids, token_id] , dim=-1)
                attention_mask = torch.concat([attention_mask, torch.ones_like(token_id)] , dim=-1)

        rollout_logprobs = torch.stack(rollout_logprobs, dim=-1).view(-1)
        response_ids = torch.stack(response_ids, dim=-1).view(-1)
        text = self._tokenizer.decode(response_ids)
        return {'text': text, 'response_ids': response_ids, 'rollout_logprobs': rollout_logprobs, 'prompt_ids': prompt_ids}



if __name__ == '__main__':
    policy = HFCausalPolicy(model_name='openai-community/gpt2', device='cpu')
    results = policy.generate('''<SYSTEM>
You are a careful math problem solver. Think step by step when useful, and end with a clear final answer.

Follow these rules strictly:
1) Solve the question and return exactly one final answer wrapped in \boxed{...}.
2) In \boxed{...}, output only the final value/expression (no words, units, punctuation, or extra spaces).
3) Never output multiple boxed answers.

<USER>
17 + 28 = ?''', 10, 0.5)

