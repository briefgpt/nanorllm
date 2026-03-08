from nanorllm.core.trajectory import Trajectory, Step

from collections import defaultdict

def group_by_task_id(
    trajectories: list[Trajectory],
) -> dict[str, list[Trajectory]]:
    
    task_grouped = defaultdict(list)
    for t in trajectories:
        task_grouped[t.task_id].append(t)
    return task_grouped

def compute_advantage(
    grouped_trajectories: dict[str, list[Trajectory]],
) -> dict[str, list[tuple[Trajectory, float]]]:
    
    advs = defaultdict(list)
    for task_id, g in grouped_trajectories.items():
        group_score = [t.final_reward for t in g]
        avg_reward = sum(group_score)/ len(group_score)
        for t in g:
            adv = t.final_reward-avg_reward
            advs[task_id].append((t, adv))
    return advs

        





def build_train_samples(
    grouped_advantages: dict[str, list[tuple[Trajectory, float]]],
) -> list[dict]:
    samples = []
    for task_id, trajectory_pairs in grouped_advantages.items():
        for trajectory, advantage in trajectory_pairs:
            for step in trajectory.steps:
                if step.model_response is None or step.prompt_messages is None:
                    continue
                sample = {'prompt_messages': step.prompt_messages, 'response': step.model_response,'advantage': advantage, 'task_id': task_id}
                samples.append(sample)
    return samples
            
