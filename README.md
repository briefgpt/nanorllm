# nanorllm

最小化复现 `agentic RL` 的核心闭环，主 demo 固定为 `multi-turn math self-refine`。

当前目标不是复现完整 `rLLM`，而是把这条链路压缩到足够小、足够清楚：

题目 -> 回答 -> 环境反馈 -> 再回答 -> terminal reward -> grouped advantage -> train samples

## 当前状态

目前已经跑通的部分：

- 多轮 `agent-env` 交互
- cumulative message history
- `Trajectory / Step` 记录
- terminal reward
- 同题多采样后的 `group_by_task_id`
- 基于 `final_reward` 的组内 advantage
- 将 trajectory 展开成 step-level train samples
- `sample -> tokenize -> response_mask -> logprob -> loss -> backward -> optimizer step`
- 一个最小 `trainer.run_train_epoch(...)` 训练入口

当前还没有做的部分：

- old policy / ratio / clip
- critic
- async rollout
- 通用化的 trainer config

## 当前目录

```text
nanorllm/
  nanorllm/
    core/
      trajectory.py
      types.py
    agents/
      base.py
      math_agent.py
    envs/
      base.py
      math_env.py
    llm/
      gemini.py
    rollout/
      engine.py
    datasets/
      simple_math.py
    algos/
      grpo.py
    policy/
      base.py
      hf_causal.py
    trainer/
      collate.py
      loss.py
      trainer.py
  examples/
    run_math_episode.py
    train_math_grpo.py
```

## 关键对象

`Step`

- 一轮 `observation -> model_response/action -> env feedback` 的记录单元
- 最小字段包括 `observation`、`prompt_messages`、`model_response`、`action`、`reward`、`done`、`info`

`Trajectory`

- 一个完整 episode 的容器
- 最小字段包括 `task_id`、`steps`、`final_reward`、`terminated`、`termination_reason`

`MathAgent`

- 维护 messages
- 接收 env observation
- 写入 `Trajectory`
- 不负责 reward 判断

`MathEnv`

- 提供题目
- 判断回答是否正确
- 错误时给 retry feedback
- 控制 episode 结束

`RolloutEngine`

- 驱动 agent-env-llm 循环
- 返回一条 `Trajectory`

`GRPO-lite`

- 输入 `list[Trajectory]`
- 按 `task_id` 分组
- 按 `final_reward` 计算组内 advantage
- 输出 step-level train samples

`Trainer`

- 负责批量 rollout
- 调用 `group_by_task_id -> compute_advantage -> expand_step_samples`
- 负责把 sample 组装成 batch，并执行一个最小 `optimizer.step()`

## 任务格式

当前 task schema 固定为：

```python
{
    "task_id": "gsm8k-001",
    "question": "...",
    "answer": "42",
}
```

环境规则固定为：

- `reset(task)` 返回 `{"question": ...}`
- 回答正确：`done=True, reward=1.0`
- 回答错误且未超轮数：返回 retry feedback，`done=False, reward=0.0`
- 最后一轮仍错误：`done=True, reward=0.0`

## 运行方式

先确保：

- 已创建虚拟环境 `.venv`
- 已安装依赖
- 已设置 `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`

单题 rollout：

```bash
.venv/bin/python examples/run_math_episode.py
```

最小 trainer train-step：

```bash
.venv/bin/python examples/train_math_grpo.py
```

如果后面新增包内执行入口，优先用 `python -m ...` 的方式运行，避免直接跑包内文件时遇到 import 路径问题。

## Trainer 数据流

当前 `trainer.run_epoch(...)` 的最小 rollout 流程是：

```python
trajectories = [rollout_fn(task) ...]
grouped = group_by_task_id(trajectories)
grouped_advantages = compute_advantage(grouped)
samples = expand_step_samples(grouped_advantages)
```

其中：

- `group_by_task_id` 输入 `list[Trajectory]`
- `compute_advantage` 输入 `dict[str, list[Trajectory]]`
- `expand_step_samples` 输入 `dict[str, list[tuple[Trajectory, float]]]`

每个 train sample 当前最小结构为：

```python
{
    "task_id": "...",
    "prompt_messages": [...],
    "response": "...",
    "advantage": 0.5,
}
```

这里的 `response` 指的是某个 step 的 `model_response`，不是整条 trajectory 的拼接，也不是单独抽取的 final answer。

当前 `trainer.run_train_epoch(...)` 的最小训练流程是：

```python
trajectories = collect_rollouts(tasks, num_samples_per_task, rollout_fn)
samples = build_step_samples_from_trajectories(trajectories)
batch = build_train_batch(samples, tokenizer=tokenizer, max_length=max_length)
outputs = policy.forward(batch["input_ids"], batch["attention_mask"])
loss, sequence_logprobs = compute_policy_loss(
    outputs.logits,
    batch["labels"],
    batch["response_mask"],
    batch["advantages"],
)
```

## 当前边界

v0 只做这些：

- 单机
- 同步 rollout
- 单一 demo: math self-refine
- terminal reward
- 2-5 轮内结束
- trajectory-level GRPO-lite
- 最小训练闭环和单步参数更新

v0 明确不做这些：

- Ray
- verl
- vLLM 深度集成
- tool use
- search
- critic
- stepwise advantage
- async rollout

## 当前训练闭环

当前已经打通的最小闭环是：

题目 -> rollout -> grouped advantage -> train samples -> tokenize -> response logprob -> policy loss -> optimizer step

当前训练侧的最小职责分工：

`nanorllm/policy/base.py`

- 定义最小训练 policy 接口

`nanorllm/policy/hf_causal.py`

- 用 `transformers` 加载可训练的 causal LM 和 tokenizer

`nanorllm/trainer/collate.py`

- 把 `expand_step_samples(...)` 产出的 sample 变成模型输入
- 明确哪些 token 属于 response，需要参与 loss

`nanorllm/trainer/loss.py`

- 从 `logits` 和 `labels` 提取 response token 的 logprob
- 结合 advantage 计算最小 policy loss

`nanorllm/trainer/trainer.py`

- 组织 rollout、sample 构造、batch 构造、前向、反传和参数更新

### 每个函数要解决的问题

`render_prompt_messages(...)`

- 把当前 `sample["prompt_messages"]` 渲染成稳定文本
- 目标不是通用 chat template，而是先把当前 math demo 喂进本地模型

`tokenize_sample(...)`

- 把 `prompt + response` 变成 `input_ids`
- 同时保留 “prompt 到哪里结束” 这个边界信息

`build_response_mask(...)`

- 只让 response 部分参与 loss
- 这是训练阶段最关键的边界函数

`compute_token_logprobs(...)`

- 从 `logits` 提取每个 label token 的对数概率

`masked_sequence_logprobs(...)`

- 只聚合 response token 的 logprob
- 输出 shape 最好是 `[batch]`

`compute_policy_loss(...)`

- 当前输入是 `logits`、`labels`、`response_mask`、`advantages`
- 内部先计算 token-level logprob，再聚合成 sequence-level logprob
- 最小版本核心仍然是：

```python
loss = -(advantages * sequence_logprobs).mean()
```

- 返回 `(loss, sequence_logprobs)`

`train_step(...)`

- 前向
- 算 loss
- backward
- optimizer step
- 返回一组简单标量，例如 `loss`、`avg_advantage`、`avg_response_logprob`

### 当前阶段的边界

当前只做这些：

- 本地可训练 causal LM
- sample 到 loss 的最小链路
- 一个 batch 的前向、反传和参数更新
- 一个 epoch 的最小训练入口

当前先不做这些：

- old policy
- ratio / clip
- reference model / KL
- checkpoint 管理
- 多卡
- rollout 和 training 解耦成两套复杂 worker

### Optimizer 相关

当前阶段的 optimizer 先保持最小化，先不做这些：

- scheduler
- weight decay 分组
- gradient clipping
- mixed precision 优化
- gradient accumulation

## 未来一周实现计划

未来一周按下面的优先级推进：

1. 先把当前 `GRPO-lite` 训练闭环补到更稳定的 PPO/GRPO 形态。
2. 再把 `trajectory-level final reward` 复制式 credit assignment，升级成 step-level credit assignment。
3. 同时补最小 eval 和日志，确保训练结果可判断。
4. 最后补一个更像 agent 的 toy 环境，而不只是多轮文字自修正。

### P0: 稳定训练闭环

本周最先完成这些：

- 保存 rollout 时的 `old_logprob`
- 在 loss 中加入 `ratio`
- 增加最小 `clip` 版本
- 增加 reference policy 或最小 KL 约束
- 增加 reward / advantage 的基本归一化
- 记录 `loss`、`avg_reward`、`avg_advantage`、`avg_response_logprob`、`approx_kl`

目标：

- 当前训练不再只是 `loss = -(advantage * logprob).mean()`
- 训练指标开始能反映 policy 是否偏移过快

### P1: credit assignment 升级

在稳定闭环后，优先做这些：

- 明确区分 `step reward`、`final reward`、`advantage`
- 不再把同一个 trajectory advantage 原样复制给每个 step
- 先实现一个最小 step-level credit assignment 版本
- 如果 step reward 还不够稳定，先做一个可解释的 heuristic 分配版本

目标：

- 让每个 step 的更新更接近它对最终结果的真实贡献
- 为后续 agentic RL 的长时程决策打基础

### P2: eval 和日志

本周至少补这些可观测性：

- 固定 train / eval task split
- 每次训练后跑一小组固定 eval tasks
- 记录 success rate、avg reward、trajectory length、response length
- 记录 step 数分布、终止原因、advantage 分布

目标：

- 能区分“训练真的提升了”还是“reward hack / sampling 波动”

### P3: 更像 agent 的 toy 环境

如果前面两项推进顺利，本周后半段开始补：

- 状态会变化的环境反馈
- 更明确的 `observation -> action -> feedback` 边界
- 非法动作或无效动作惩罚
- 更清楚的终止条件和 `termination_reason`

目标：

- 让当前 demo 从 `multi-turn math self-refine` 更接近真正的 `agent-env loop`

### 本周完成标志

这一周结束时，对应的 v0.1 目标是：

- 有一个带 `old_logprob + ratio/clip + KL` 的最小训练版本
- 有一个 step-level credit assignment 的最小实现
- 有一组固定 eval 指标可持续追踪
- 有一个比当前 math demo 更接近 agent 行为的 toy 环境原型

## 备注

当前更看重 demo 能跑通、数据流清楚、对象职责明确。

当前已经补了最小单测，优先锁这些不变量：

- 单题 episode 行为符合预期
- batch rollout 能形成同题分组
- advantage 和 train sample 组织正确
- sample 到 loss 的边界清楚
