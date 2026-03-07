from nanorllm.agents.math_agent import MathAgent
from nanorllm.envs.math_env import MathEnv
from nanorllm.rollout.engine import RolloutEngine


class AlwaysWrongLLM:
    def generate(self, messages):
        return "999"


def test_engine_max_steps_fuse_marks_terminated_with_reason():
    task = {"task_id": "t-fuse", "question": "1+1-2=?", "ground_truth": "0"}
    agent = MathAgent(system_prompt="Answer the math question.")
    env = MathEnv()
    env.max_turn = 10
    llm = AlwaysWrongLLM()
    engine = RolloutEngine()

    max_steps = 3
    trajectory = engine.run_episode(agent, env, llm, task, max_steps=max_steps)

    assert trajectory.terminated is True
    assert trajectory.termination_reason == "max step"
    assert sum(1 for step in trajectory.steps if step.model_response is not None) == max_steps

