class RolloutEngine:
    def run_episode(self, agent, env, llm, task, max_steps=10):
        agent.reset()
        observation, info = env.reset(task)
        reward = 0.0
        done = False
        agent.update_from_env(observation, reward, done, info)

        for i in range(max_steps):
            response = llm.generate(agent.messages)
            action = agent.update_from_model(response)
            observation, reward, done, info = env.step(action)
            print(i, info, done)
            agent.update_from_env(observation, reward, done, info)
            if done:
                agent.trajectory.terminated=True
                break
        if not agent.trajectory.terminated:
            agent.trajectory.termination_reason = 'max step'
            agent.trajectory.terminated = True
        return agent.trajectory
