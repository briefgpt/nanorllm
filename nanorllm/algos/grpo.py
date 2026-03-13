from collections import defaultdict

from nanorllm.core.trajectory import StepSample, RolloutResult



def group_by_task_id(episode_outputs: list[RolloutResult]) -> dict[str | None, list[RolloutResult]]:
    grouped_episode_outputs: dict[str | None, list[RolloutResult]] = defaultdict(list)
    for rollout in episode_outputs:
        grouped_episode_outputs[rollout.trajectory.task_id].append(rollout)
    return grouped_episode_outputs


def compute_advantage(
    grouped_episode_outputs: dict[str | None, list[RolloutResult]],
) -> dict[str | None, list[RolloutResult]]:
    for task_id, g in grouped_episode_outputs.items():
        group_score = [rollout.trajectory.final_reward for rollout in g]
        avg_reward = sum(group_score) / len(group_score)
        for rollout in g:
            adv = rollout.trajectory.final_reward - avg_reward
            for sample in rollout.episode_step_samples:
                sample.advantage = adv
    return grouped_episode_outputs


def flatten_step_samples(
    grouped_episode_outputs: dict[str | None, list[RolloutResult]],
) -> list[StepSample]:
    steps: list[StepSample] = []
    for task_id, episodes in grouped_episode_outputs.items():
        for rollout in episodes:
            steps.extend(rollout.episode_step_samples)
    return steps
        

        





