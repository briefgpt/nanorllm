from nanorllm.envs.base import BaseEnv
from nanorllm.core.types import Action, RewardOutput
import re


def normalize_math_answer(text):
    text = text.strip()
    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_matches:
        text = boxed_matches[-1].strip()
    return text


class MathEnv(BaseEnv):
    def __init__(self):
        self.task = None
        self.turn_count = 0
        self.max_turn = 5

    def reset(self, task):
        self.task = task
        observation = {"question": task['question']}
        info = {"task_id": task['task_id']}
        self.turn_count = 0 # 不能忘
        return observation, info
    
    def step(self, action):
        origin_prediction = action.value
        prediction = normalize_math_answer(action.value)
        judge_res = self.verifier(prediction, self.task)
        self.turn_count += 1
        normalize_math_answer(action.value)
        if judge_res:
            done = True
            reward = 1.0
            observation = {"feedback": "success"}
            info = self._build_info(prediction, origin_prediction, termination_reason= 'env_done')
        else:
            if self.turn_count < self.max_turn:
                reward = 0.0
                done = False
                observation = {"feedback": "Your previous answer is incorrect. Try again and put the final answer clearly."}
                info = self._build_info(prediction, origin_prediction, termination_reason= 'continue try')

            else:
                reward = 0.0
                done = True
                observation = {"feedback": "exceeds max turn"}
                info = self._build_info(prediction, origin_prediction, termination_reason= 'max_turn')
            

        return observation, reward, done, info
    
    def _build_info(self, prediction, origin_prediction, termination_reason):
        return {'prediction': prediction, 'origin_prediction': origin_prediction, 'termination_reason': termination_reason}

    def verifier(self, prediction, task):
        if  prediction == task['answer']:
            return True
        else:
            return False


