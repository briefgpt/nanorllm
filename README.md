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
- 一个最小 `trainer.run_epoch(...)` dry-run

当前还没有做的部分：

- 真正的 policy gradient / 反传更新
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
- 调用 `group_by_task_id -> compute_advantage -> build_train_samples`
- 当前只做 dry-run，不做参数更新

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

最小 trainer dry-run：

```bash
.venv/bin/python examples/train_math_grpo.py
```

如果后面新增包内执行入口，优先用 `python -m ...` 的方式运行，避免直接跑包内文件时遇到 import 路径问题。

## Trainer 数据流

当前 `trainer.run_epoch(...)` 的最小流程是：

```python
trajectories = [rollout_fn(task) ...]
grouped = group_by_task_id(trajectories)
grouped_advantages = compute_advantage(grouped)
samples = build_train_samples(grouped_advantages)
```

其中：

- `group_by_task_id` 输入 `list[Trajectory]`
- `compute_advantage` 输入 `dict[str, list[Trajectory]]`
- `build_train_samples` 输入 `dict[str, list[tuple[Trajectory, float]]]`

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

## 当前边界

v0 只做这些：

- 单机
- 同步 rollout
- 单一 demo: math self-refine
- terminal reward
- 2-5 轮内结束
- trajectory-level GRPO-lite
- trainer dry-run

v0 明确不做这些：

- Ray
- verl
- vLLM 深度集成
- tool use
- search
- critic
- stepwise advantage
- async rollout

## 下一阶段：真实训练版

下一阶段的目标不是继续扩 rollout，而是把当前的 `train samples` 真正接到一个可训练的本地 policy 上。

新的最小闭环应变成：

题目 -> rollout -> grouped advantage -> train samples -> tokenize -> response logprob -> policy loss -> optimizer step

### 推荐新增文件

`nanorllm/policy/base.py`

- 定义最小训练 policy 接口
- 只服务训练，不替代当前的 Gemini rollout backend

建议函数：

- `build_policy(model_name: str, device: str) -> BasePolicy`
- `generate_text(...)` 可以先不做，训练阶段不必复用 rollout 生成

`nanorllm/policy/hf_causal.py`

- 用 `transformers` 加载一个可训练的 causal LM 和 tokenizer
- 提供前向计算所需的最小包装

建议函数：

- `load_tokenizer(model_name: str)`
- `load_model(model_name: str, device: str)`
- `forward(input_ids, attention_mask) -> logits`

`nanorllm/trainer/collate.py`

- 负责把 `build_train_samples(...)` 产出的 sample 变成模型输入
- 这一步要明确“哪些 token 属于 response，需要被优化”

建议函数：

- `render_prompt_messages(prompt_messages: list[dict]) -> str`
- `format_training_text(sample: dict) -> dict[str, str]`
- `tokenize_sample(sample: dict, tokenizer, max_length: int) -> dict[str, torch.Tensor]`
- `build_response_mask(input_ids, prompt_input_ids_len: int) -> torch.Tensor`
- `collate_train_batch(samples: list[dict], tokenizer, max_length: int) -> dict[str, torch.Tensor]`

`nanorllm/trainer/loss.py`

- 负责把模型输出和 advantage 组装成最小 policy loss
- v1 不碰 old policy / ratio / clip，先做 advantage-weighted logprob

建议函数：

- `compute_token_logprobs(logits, labels) -> torch.Tensor`
- `masked_sequence_logprobs(token_logprobs, response_mask) -> torch.Tensor`
- `compute_policy_loss(sequence_logprobs, advantages) -> torch.Tensor`
- `summarize_batch_metrics(sequence_logprobs, advantages, loss) -> dict[str, float]`

`nanorllm/trainer/trainer.py`

- 在当前 `run_epoch(...)` 基础上，继续补训练侧 orchestration
- 区分 `collect_rollouts`、`build_train_batch`、`train_step`

建议函数：

- `collect_rollouts(tasks, num_samples_per_task, rollout_fn) -> list[Trajectory]`
- `build_training_samples(trajectories) -> list[dict]`
- `train_step(model, optimizer, batch) -> dict[str, float]`
- `run_train_epoch(tasks, num_samples_per_task, rollout_fn, model, tokenizer, optimizer) -> dict`

`examples/train_math_grpo.py`

- 作为真实训练版入口
- 先打印 batch 统计，再执行一个最小 `optimizer.step()`

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

- 最小版本直接做：

```python
loss = -(advantages * sequence_logprobs).mean()
```

- 先不要加 ratio、clip、KL

`train_step(...)`

- 前向
- 算 loss
- backward
- optimizer step
- 返回一组简单标量，例如 `loss`、`avg_advantage`、`avg_response_logprob`

### 建议实现顺序

1. `render_prompt_messages`
2. `tokenize_sample`
3. `build_response_mask`
4. `compute_token_logprobs`
5. `masked_sequence_logprobs`
6. `compute_policy_loss`
7. `train_step`
8. `run_train_epoch`

### 当前阶段的边界

下一阶段只做这些：

- 本地可训练 causal LM
- sample 到 loss 的最小链路
- 一个 batch 的前向和反传
- 一个 epoch 的最小训练入口

下一阶段先不做这些：

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

## 备注

当前更看重 demo 能跑通、数据流清楚、对象职责明确。

测试后续再补；现阶段优先保证：

- 单题 episode 行为符合预期
- batch rollout 能形成同题分组
- advantage 和 train sample 组织正确
- sample 到 loss 的边界清楚
