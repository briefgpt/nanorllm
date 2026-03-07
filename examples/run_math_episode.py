import argparse
import sys
from pathlib import Path

# Allow running `python3 examples/run_math_episode.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanorllm.rollout.engine import RolloutEngine
from nanorllm.agents.math_agent import MathAgent
from nanorllm.envs.math_env import MathEnv
from nanorllm.llm.gemini import GeminiLLM

demo_data = {"question": "80 增加 15% 后是多少？", "ground_truth": "92", "task_id": "000"}


system_prompt = """You are a careful math problem solver. Think step by step when useful, and end with a clear final answer.

Follow these rules strictly:
1) Solve the question and return exactly one final answer wrapped in \\boxed{...}.
2) In \\boxed{...}, output only the final value/expression (no words, units, punctuation, or extra spaces).
3) Never output multiple boxed answers.
"""




def run_math_episode( max_steps: int = 10, max_turn: int = 2):
    agent = MathAgent(system_prompt=system_prompt)
    env = MathEnv()
    env.max_turn = max_turn
    llm = GeminiLLM(timeout=200)

    engine = RolloutEngine()
    trajectory = engine.run_episode(agent, env, llm, demo_data, max_steps=max_steps)
    return trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-turn", type=int, default=5)
    args = parser.parse_args()

    traj = run_math_episode( max_steps=args.max_steps, max_turn=args.max_turn)
    print(f"steps={len(traj.steps)} final_reward={traj.final_reward} terminated={traj.terminated} reason={traj.termination_reason}")
