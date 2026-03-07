from nanorllm.agents.math_agent import MathAgent
from nanorllm.envs.math_env import MathEnv
from nanorllm.rollout.engine import RolloutEngine


class CorrectFirstTryLLM:
    def generate(self, messages):
        return "0"


def test_agent_reset_clears_messages_and_trajectory_for_new_episode():
    task = {"task_id": "t-agent", "question": "1+1-2=?", "ground_truth": "0"}
    system_prompt = "Answer the math question."
    agent = MathAgent(system_prompt=system_prompt)
    env = MathEnv()
    llm = CorrectFirstTryLLM()
    engine = RolloutEngine()

    trajectory = engine.run_episode(agent, env, llm, task, max_steps=3)
    assert trajectory.terminated is True
    assert len(trajectory.steps) >= 1

    agent.reset()

    assert agent.messages == [{"role": "system", "content": system_prompt}]
    assert agent.trajectory.steps == []
    assert agent.trajectory.task_id is None
    assert agent.trajectory.final_reward == 0.0
    assert agent.trajectory.terminated is False
    assert agent.trajectory.termination_reason is None

