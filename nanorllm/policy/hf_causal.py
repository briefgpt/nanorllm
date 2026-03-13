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

    def _sample_token(self, logits: torch.Tensor, temperature: float):
        scaled_logits = logits / temperature
        probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1)
        log_probs = torch.nn.functional.log_softmax(scaled_logits, dim=-1)
        token_log_prob = torch.gather(log_probs, index=token_id, dim=-1)
        return token_id, token_log_prob

    def generate(self, prompt_text: str, args):
        response_ids = []
        rollout_logprobs = []

        tokenized_prompt = self._tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        prompt_ids = tokenized_prompt["input_ids"]
        input_ids = prompt_ids.to(self.device)
        attention_mask = tokenized_prompt["attention_mask"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values # kv cache

            for _ in range(args.max_new_tokens):
                token_id, token_log_prob = self._sample_token(logits, args.temperature)
                rollout_logprobs.append(token_log_prob)
                response_ids.append(token_id)

                if token_id.item() == self._tokenizer.eos_token_id:
                    break

                attention_mask = torch.concat(
                    [attention_mask, torch.ones_like(token_id)],
                    dim=-1,
                )
                outputs = self.model(
                    input_ids=token_id,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

        if response_ids:
            rollout_logprobs = torch.stack(rollout_logprobs, dim=-1).view(-1)
            response_ids = torch.stack(response_ids, dim=-1).view(-1)
        else:
            rollout_logprobs = torch.empty(0, device=self.device)
            response_ids = torch.empty(0, dtype=torch.long, device=self.device)

        response_ids = response_ids.detach().cpu()
        rollout_logprobs = rollout_logprobs.detach().cpu()
        prompt_ids = prompt_ids.view(-1).detach().cpu()

        text = self._tokenizer.decode(response_ids)
        return {
            "text": text,
            "response_ids": response_ids,
            "rollout_logprobs": rollout_logprobs,
            "prompt_ids": prompt_ids,
        }



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
