from nanorllm.algos.grpo import  compute_advantage, group_by_task_id, flatten_step_samples
from nanorllm.core.trajectory import StepSample, RolloutResult
from nanorllm.trainer.collate import collate_train_batch
from nanorllm.trainer.loss import (
    compute_policy_loss,
    summarize_batch_metrics,
)


def collect_rollouts(tasks, num_samples_per_task, rollout_fn) -> list[RolloutResult]:
    """
    Collect multi-sample rollout results for each task.

    Each episode output keeps the trajectory-level semantics and the step-level
    training samples together as `(trajectory, episode_step_samples)`.
    """
    episode_outputs = []
    for task in tasks:
        for _ in range(num_samples_per_task):
            rollout_result = rollout_fn(task)
            episode_outputs.append(rollout_result)
    return episode_outputs


def build_step_samples_from_episode_outputs(
    episode_outputs: list[RolloutResult],
) -> list[StepSample]:
    grouped_episode_outputs = group_by_task_id(episode_outputs)
    grouped = compute_advantage(grouped_episode_outputs)
    samples = flatten_step_samples(grouped)
    return samples



def train_step(policy, optimizer, batch, args):
    policy.model.train()
    optimizer.zero_grad()
    outputs = policy.forward(batch['input_ids'], batch['attention_mask'])
    logits = outputs.logits
    loss = compute_policy_loss(logits, batch, args)
    loss.backward()
    optimizer.step()
    metrics = summarize_batch_metrics( batch['advantages'], loss)
    return metrics



def run_train_epoch(
    tasks,
    rollout_fn,
    policy,
    tokenizer,
    optimizer,
    args
):
    episode_outputs = collect_rollouts(tasks, args.num_samples_per_task, rollout_fn)
    samples = build_step_samples_from_episode_outputs(episode_outputs)
    batch = collate_train_batch(samples, tokenizer, args)
    metrics = train_step(policy=policy, optimizer=optimizer, batch=batch, args=args)
    trajectories= [rollout.trajectory for rollout in episode_outputs]
    return {
        "episode_outputs": episode_outputs,
        "trajectories": trajectories,
        "samples": samples,
        "batch": batch,
        "metrics": metrics,
    }
