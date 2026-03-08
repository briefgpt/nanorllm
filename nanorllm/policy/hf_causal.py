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
