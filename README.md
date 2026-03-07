学习Agentic RL，复现rLLM 最小实现版本


把 multi-turn math self-refine 作为 nanorllm 的主 demo 后，整个复现计划会清晰很多，因为你只需要复现 agentic RL 的核心闭环：

题目 -> 回答 -> 环境反馈 -> 再回答 -> 最终 reward -> policy update

目标重定义

你要复现的不是 rllm 全仓库，而是一个最小可工作的 nanorllm：

支持多轮 agent-env 交互
支持 cumulative message history
支持 trajectory 记录
支持 terminal reward
支持同题多采样的 GRPO-lite
demo 任务固定为 math self-refine
v0 最终效果

你做完后，应该能跑这样一个例子：

env 给一道数学题
agent 第一次作答
env 判断是否正确
如果错误，env 返回反馈
agent 再答一次
最多 2-3 轮
记录完整 trajectory
对同一题采样多条轨迹，按 reward 算 advantage
做一次最小 policy update 或至少把训练数据整理出来
实现边界

v0 只做这些：

单机
同步 rollout
单一 demo: math self-refine
terminal reward
最多 2-3 轮
trajectory-level GRPO-lite
v0 不做这些：

Ray
verl
vLLM 深度集成
tool use
search
critic
stepwise advantage
async rollout
推荐的 4 个阶段

阶段 1：先把交互闭环跑通

目标：不训练，只能稳定跑一条 episode。

你先实现这几个文件：

nanorllm/core/types.py
nanorllm/core/trajectory.py
nanorllm/agents/base.py
nanorllm/agents/math_agent.py
nanorllm/envs/base.py
nanorllm/envs/math_self_refine.py
nanorllm/llm/base.py
nanorllm/llm/gemini.py
nanorllm/rollout/engine.py
examples/run_math_episode.py
这阶段要做的事：

定义 Step / Trajectory
定义 BaseAgent / BaseEnv
写一个 MathAgent
写一个 MathSelfRefineEnv
写 run_episode
只维护一套真实模型调用链路（例如 Gemini 或 OpenAI-compatible）；测试里可用 stub/mock，不新增第二套 LLM 实现模块。

验收标准：

python examples/run_math_episode.py 能输出完整 trajectory
trajectory 里能看到多轮 observation / response / reward / done
阶段 2：把 math self-refine env 做对

这是整个 demo 最关键的部分。

环境规则建议固定成：

reset(task) 返回：
{"question": ...}
第一次回答后：
如果正确，done=True, reward=1
如果错误且未超轮数，返回：
"Your previous answer is incorrect. Try again and put the final answer clearly."
最后一轮还错：
done=True, reward=0
task 格式先定死成：

{
    "task_id": "gsm8k-001",
    "question": "...",
    "answer": "42",
}
建议最大轮数：

max_turns = 2 或 3
验收标准：

正确答案直接结束
错误答案会收到反馈并进入下一轮
超过最大轮数会结束
reward 只在结束时给出
阶段 3：扩展 LLM backend 和小数据集

目标：让 demo 真正“像个 agent”。

新增文件：

nanorllm/llm/openai_chat.py
nanorllm/rewards/exact_match.py
nanorllm/datasets/simple_math.py
examples/run_math_eval.py
这阶段要做的事：

在已有真实模型链路基础上，新增一个 OpenAI-compatible chat backend
准备 20-100 道小 math 数据
跑 inference，不训练
看 self-refine 是否真的能改善结果
你此时要观察的不是分数，而是：

第一轮错的时候，第二轮是否会修正
prompt 累积是否自然
trajectory 是否可读
验收标准：

能批量跑数据
每题能保存 trajectory
至少部分题能通过第二轮修正
阶段 4：加最小训练

新增文件：

nanorllm/algos/grpo.py
nanorllm/trainer/config.py
nanorllm/trainer/trainer.py
examples/train_math_grpo.py
训练目标不要贪大，只做：

同一题采样 n 条轨迹
按最终 reward 做组内 advantage
trajectory-level GRPO-lite
最小训练循环：

取一批题
每题 rollout n 次
收集 trajectories
按 task_id 分组
根据 final reward 算 advantage
对回答 token 做 policy gradient update
这阶段如果你暂时不想碰真正反传，也可以先做“伪训练版”：

先把 GRPO training batch 整理出来
先验证 advantage 和样本组织没问题
验收标准：

能输出 grouped trajectories
每个 task 的 advantage 计算正确
trainer 能完成一个 epoch
建议的目录结构

nanorllm/
  nanorllm/
    core/
      types.py
      trajectory.py
    agents/
      base.py
      math_agent.py
    envs/
      base.py
      math_self_refine.py
    llm/
      base.py
      gemini.py
      openai_chat.py
    rewards/
      exact_match.py
    rollout/
      engine.py
    algos/
      grpo.py
    trainer/
      config.py
      trainer.py
    datasets/
      simple_math.py
  examples/
    run_math_episode.py
    run_math_eval.py
    train_math_grpo.py
  tests/
    test_trajectory.py
    test_math_env.py
    test_rollout_engine.py
    test_grpo.py
核心模块怎么分工

MathAgent

维护 messages
接收 env observation
把 model response 写进 trajectory
不做 reward 判断
MathSelfRefineEnv

提供题目
判断回答是否正确
错误时给 retry feedback
控制 episode 结束
RolloutEngine

驱动 agent-env-llm 循环
返回 trajectory
GRPO

输入 trajectories
输出 grouped advantages 和 loss
Trainer

负责 sample / train / eval orchestration
你动手实现时的顺序

Step / Trajectory
MathSelfRefineEnv
MathAgent
run_episode
单题联调（真实模型链路）
小样本数据集
OpenAIChatLLM
批量 rollout
GRPO-lite
trainer
这个顺序不要改。先写 trainer 很容易空转。

第一周的具体计划

第 1 天
实现 core + env + agent，跑通真实模型链路下的单题两轮交互

第 2 天
实现 run_episode、trajectory logging、基础测试

第 3 天
接真实 LLM，跑 20 道题，手动检查 trajectory

第 4 天
做 batch rollout，按 task_id 组织 grouped samples

第 5 天
实现 GRPO-lite 的 advantage 计算

第 6-7 天
补 trainer 骨架，跑一个最小训练闭环

最重要的 3 个实现原则

先把 trajectory 打印清楚，再考虑训练
先把 env 规则写死，再谈抽象
先做 terminal reward，再做更复杂的 reward shaping
我建议你现在立即开始写的第一个文件

先写 nanorllm/envs/math_self_refine.py。
因为一旦 env 规则定下来，agent、rollout、trainer 都会自然落位。
