import sys
from pathlib import Path
import os
import torch
from dataclasses import dataclass

# Allow running `python3 examples/run_math_episode.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from nanorllm.rollout.engine import RolloutEngine
from nanorllm.agents.math_agent import MathAgent
from nanorllm.envs.math_env import MathEnv
from nanorllm.datasets.simple_math import get_simple_math_tasks
from nanorllm.trainer.trainer import run_train_epoch
from nanorllm.policy.hf_causal import HFCausalPolicy

"""
cd /Users/sl/caitian/nanorllm
source .venv/bin/activate
python examples/train_math_grpo.py
"""


system_prompt = """You are a careful math problem solver. Think step by step when useful, and end with a clear final answer.

Follow these rules strictly:
1) Solve the question and return exactly one final answer wrapped in \\boxed{...}.
2) In \\boxed{...}, output only the final value/expression (no words, units, punctuation, or extra spaces).
3) Never output multiple boxed answers.
"""


@dataclass
class TrainArgs:
    clip_eps: float=0.2
    use_kl: bool=False
    kl_coef: float=0.001

    temperature: float = 1.0
    max_new_tokens: int=12
    max_steps: int = 5
    num_samples_per_task: int = 2
    max_length: int = 1024

    lr: float = 1e-5

args = TrainArgs()
engine = RolloutEngine()
agent = MathAgent(system_prompt=system_prompt)
env = MathEnv()
policy = HFCausalPolicy(model_name='openai-community/gpt2', device='cpu')
tokenizer = policy._tokenizer
optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

def rollout_fn(task):
    return engine.run_episode(agent, env, policy, task, args)



tasks = get_simple_math_tasks()[:2]


result = run_train_epoch(
    tasks,
    rollout_fn,
    policy,
    tokenizer,
    optimizer,
    args
)
print(result['batch'])
print(result['trajectories'])
