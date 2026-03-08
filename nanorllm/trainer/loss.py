import torch
import torch.nn.functional as F


def compute_token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    log_probs = F.log_softmax(shifted_logits, dim=-1) # 注意这里是log_softmax
    token_probs = torch.gather(log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
    return token_probs


def masked_sequence_logprobs(
    token_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    mask = response_mask[:, 1:].to(token_logprobs.dtype) # 注意这里也需要shift，否则和token_logprobs 长度对不上，和labels的切片对齐
    masked_sequence_logprobs = token_logprobs* mask
    return masked_sequence_logprobs.sum(-1)/ mask.sum(-1)



def compute_policy_loss(
    sequence_logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    return -(advantages * sequence_logprobs).mean()


def summarize_batch_metrics(
    sequence_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss: torch.Tensor,
) -> dict[str, float]:
    return {
        "loss": float(loss.detach().item()),
        "avg_advantage": float(advantages.detach().mean().item()),
        "avg_response_logprob": float(sequence_logprobs.detach().mean().item()),
    }
