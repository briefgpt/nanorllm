from nanorllm.algos.grpo import compute_advantage
from nanorllm.core.trajectory import Trajectory
from nanorllm.trainer.collate import collate_train_batch
from nanorllm.trainer.loss import (
    compute_policy_loss,
    summarize_batch_metrics,
)
from collections import defaultdict




def collect_rollouts(tasks, num_samples_per_task, rollout_fn) -> list[Trajectory]:
    '''
    通过rollout_fn 收集轨迹，每个task 采样 num_samples_per_task 个rollouts
    '''
    trajectories = []
    for task in tasks:
        for _ in range(num_samples_per_task):
            trajectory = rollout_fn(task)
            trajectories.append(trajectory)
    return trajectories



def build_step_samples_from_trajectories(trajectories: list[Trajectory]) -> list[dict]:
    '''
    将收集到的轨迹转成step级训练样本，每个训练样本由以下字段组成：
    {
        'prompt_messages': step.prompt_messages, 
        'response': step.model_response,
        'advantage': advantage, 
        'task_id': task_id
    }
    1. 按照task_id 将轨迹分组
    2. 计算每个样本的每个step相对于组的优势
    3. 汇总
    '''
    samples = compute_advantage(trajectories)


    steps = []

    for task_id, trajectories in samples.items():
        for t in trajectories:
            for step in t.steps:
                steps.append({
                    'prompt_ids': step.prompt_ids,
                    'response_ids': step.response_ids,
                    'rollout_logprobs': step.rollout_logprobs,
                    'advantage': step.advantage
                })
    return steps



def train_step(policy, optimizer, batch, args):

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
    trajectories = collect_rollouts(tasks, args.num_samples_per_task, rollout_fn)
    samples = build_step_samples_from_trajectories(trajectories)
    batch = collate_train_batch(samples, tokenizer, args)
    metrics = train_step(policy=policy, optimizer=optimizer, batch=batch, args=args)
    return {
        "trajectories": trajectories,
        "samples": samples,
        "batch": batch,
        "metrics": metrics,
    }
