from nanorllm.utils.util import render_prompt_for_completion
from nanorllm.core.trajectory import StepSample, RolloutResult

class RolloutEngine:
    def run_episode(self, agent, env, llm, task, args):
        '''
        实际的rollout 过程：Agent loop
        从agent 和 env 初始化开始，通过agent.update_from_env将observation初始化到messages和trajectory里面
        loop：llm -> agent.update_from_model -> env.step(action) -> agent.update_from_env
        如果task done 或者超过max_steps 数则停止
        '''
        agent.reset()
        observation, info = env.reset(task)
        reward = 0.0
        done = False
        agent.update_from_env(observation, reward, done, info)
        episode_step_samples = []
        for i in range(args.max_steps):
            prompt_text = render_prompt_for_completion(agent.messages)
            model_output = llm.generate(prompt_text, args)
            step_sample = StepSample(prompt_ids=model_output['prompt_ids'], response_ids=model_output['response_ids'], rollout_logprobs=model_output['rollout_logprobs'])
            episode_step_samples.append(step_sample)
            action = agent.update_from_model(model_output['text'])
            observation, reward, done, info = env.step(action)
            


            agent.update_from_env(observation, reward, done, info)
            if done:
                agent.trajectory.terminated=True
                break
        if not agent.trajectory.terminated:
            agent.trajectory.termination_reason = 'max step'
            agent.trajectory.terminated = True
        return RolloutResult(trajectory=agent.trajectory, episode_step_samples=episode_step_samples)
