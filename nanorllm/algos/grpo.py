from nanorllm.core.trajectory import Trajectory, Step

from collections import defaultdict



def compute_advantage(
    trajectories: list[Trajectory],
) -> dict[str, list[Trajectory]]:
    
    grouped_trajectories = defaultdict(list)
    for t in trajectories:
        grouped_trajectories[t.task_id].append(t)

    samples = defaultdict(list)
    for task_id, g in grouped_trajectories.items():
        group_score = [t.final_reward for t in g]
        avg_reward = sum(group_score)/ len(group_score)
        for t in g:
            adv = t.final_reward-avg_reward
            for step in t.steps:
                step.advantage = adv
            samples[task_id].append(t)
    return samples
            



        






