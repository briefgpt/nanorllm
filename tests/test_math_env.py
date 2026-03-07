from nanorllm.core.types import Action
from nanorllm.envs.math_env import MathEnv


def _task():
    return {"task_id": "t-001", "question": "1+1-2=?", "ground_truth": "0"}


def test_reset_returns_question_task_id_and_resets_turn_count():
    env = MathEnv()
    env.turn_count = 99

    observation, info = env.reset(_task())

    assert observation == {"question": "1+1-2=?"}
    assert info["task_id"] == "t-001"
    assert env.turn_count == 0


def test_step_correct_answer_finishes_with_terminal_reward():
    env = MathEnv()
    env.reset(_task())

    observation, reward, done, info = env.step(Action(value="0"))

    assert done is True
    assert reward == 1.0
    assert observation["feedback"] == "success"
    assert info["termination_reason"] == "env_done"


def test_step_wrong_answer_retries_then_hits_max_turn():
    env = MathEnv()
    env.max_turn = 2
    env.reset(_task())

    observation1, reward1, done1, info1 = env.step(Action(value="1"))
    observation2, reward2, done2, info2 = env.step(Action(value="2"))

    assert done1 is False
    assert reward1 == 0.0
    assert "wrong" in observation1["feedback"].lower()
    assert info1["termination_reason"] == "continue try"

    assert done2 is True
    assert reward2 == 0.0
    assert "exceeds max turn" in observation2["feedback"].lower()
    assert info2["termination_reason"] == "max_turn"

