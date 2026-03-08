from nanorllm.algos.grpo import group_by_task_id, compute_advantage, build_train_samples
from nanorllm.core.trajectory import Trajectory
from nanorllm.trainer.collate import collate_train_batch
from nanorllm.trainer.loss import compute_policy_loss, compute_token_logprobs,masked_sequence_logprobs, summarize_batch_metrics

def run_epoch(tasks, num_samples_per_task, rollout_fn):
    trajectories = collect_rollouts(tasks, num_samples_per_task, rollout_fn)
    grouped = group_by_task_id(trajectories)
    grouped_advantages = compute_advantage(grouped)
    samples = build_train_samples(grouped_advantages)
    return {
        "trajectories": trajectories,
        "grouped": grouped,
        "grouped_adv": grouped_advantages,
        "samples": samples,
    }


def collect_rollouts(tasks, num_samples_per_task, rollout_fn) -> list[Trajectory]:
    trajectories = []
    for task in tasks:
        for _ in range(num_samples_per_task):
            trajectory = rollout_fn(task)
            trajectories.append(trajectory)
    return trajectories


def build_training_samples(trajectories: list[Trajectory]) -> list[dict]:
    grouped = group_by_task_id(trajectories)
    grouped_advantages = compute_advantage(grouped)
    return build_train_samples(grouped_advantages)


def build_train_batch(samples, tokenizer, max_length: int):
    return collate_train_batch(samples, tokenizer=tokenizer, max_length=max_length)


def train_step(policy, optimizer, batch):
    optimizer.zero_grad()
    outputs = policy.forward(batch['input_ids'], batch['attention_mask'])
    logits = outputs.logits
    token_probs = compute_token_logprobs(logits, batch['labels'])
    seq_probs = masked_sequence_logprobs(token_probs, batch['response_mask'])
    loss = compute_policy_loss(seq_probs, batch['advantages'])
    loss.backward()
    optimizer.step()
    metrics = summarize_batch_metrics(seq_probs, batch['advantages'], loss)
    return metrics



def run_train_epoch(
    tasks,
    num_samples_per_task,
    rollout_fn,
    policy,
    tokenizer,
    optimizer,
    max_length: int,
):
    trajectories = collect_rollouts(tasks, num_samples_per_task, rollout_fn)
    samples = build_training_samples(trajectories)
    batch = build_train_batch(samples, tokenizer=tokenizer, max_length=max_length)
    metrics = train_step(policy=policy, optimizer=optimizer, batch=batch)
    return {
        "trajectories": trajectories,
        "samples": samples,
        "batch": batch,
        "metrics": metrics,
    }
