from nanorllm.utils.util import render_prompt_messages


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

        for i in range(args.max_steps):
            prompt_text = render_prompt_messages(agent.messages)
            model_output = llm.generate(prompt_text, args.max_new_tokens)
            action = agent.update_from_model(model_output['text'])
            observation, reward, done, info = env.step(action)
            agent.trajectory.steps[-1].prompt_ids = model_output['prompt_ids']
            agent.trajectory.steps[-1].response_ids = model_output['response_ids']
            agent.trajectory.steps[-1].rollout_logprobs = model_output['rollout_logprobs']

            agent.update_from_env(observation, reward, done, info)
            if done:
                agent.trajectory.terminated=True
                break
        if not agent.trajectory.terminated:
            agent.trajectory.termination_reason = 'max step'
            agent.trajectory.terminated = True
        return agent.trajectory
