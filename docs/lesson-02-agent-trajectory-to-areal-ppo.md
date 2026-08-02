# 第二课：一条 Agent 轨迹怎样变成 AReaL PPO 训练数据

先给结论：Agent 完成任务时产生的是一串异构事件，PPO 不能直接训练这串事件。AReaL 会把其中每次 LLM 调用转换为 token 序列，并为每个 token 配上是否参与损失、采样时的 log-prob、采样 policy 版本和轨迹 reward。PPO 只对模型自己生成且 `loss_mask=1` 的 token 求梯度。工具返回值、系统提示、用户问题和 Harness 注入内容可以影响下一次生成，但它们作为输入 token 时不直接求梯度。

这一课要建立一条可以逐项核对的数据链。理解这条链以后，字段含义会自然对上：

```text
task
  -> session
  -> LLM call 1
  -> tool observation
  -> LLM call 2
  -> reward
  -> interaction tensors
  -> advantage
  -> PPO update
  -> policy version + 1
  -> inference 权重同步
```

## 0. 代码基线与课程边界

本课对照的是本地 AReaL 仓库的 tag `v2.0.0`，tag 指向提交：

```text
fee938eada49208a5aabdbc1095730a13076a349
```

所有 AReaL 结论均来自该 tag 下的代码，而不是把当前 `main` 的行为反推给旧版本。重点路径如下：

| 作用 | AReaL v2.0.0 路径 |
|---|---|
| 推理返回的 token 数据 | `areal/api/io_struct.py` |
| interaction 转 PPO 张量 | `areal/experimental/openai/types.py` |
| 多轮 reward 回传 | `areal/experimental/openai/cache.py` |
| session 与 trajectory 生命周期 | `areal/v2/inference_service/data_proxy/session.py` |
| 在线和离线 rollout 编排 | `areal/v2/inference_service/controller/workflow.py` |
| Agent Service 注入推理路由 | `areal/v2/agent_service/data_proxy/app.py` |
| PPO 主循环与权重同步 | `areal/trainer/rl_trainer.py` |
| advantage 与 actor loss | `areal/trainer/ppo/actor.py` |
| PPO ratio 与 clipping | `areal/utils/functional/functional.py` |
| Hermes 接入 Agent Service | `examples/hermes/hermes.py` |
| Hermes session 与 reward 脚本 | `examples/hermes/start_session.py`、`examples/hermes/set_reward.py` |
| Hermes 训练配置 | `examples/hermes/config.yaml` |

本项目的对照路径如下：

| 作用 | 本项目路径 |
|---|---|
| calculator 任务与安全求值器 | `jphrl/envs/calculator.py` |
| 13 个事件的执行链 | `jphrl/runner.py` |
| 联合版本与轨迹校验 | `jphrl/trajectory/schema.py` |
| token 元数据校验 | `jphrl/trajectory/token_contract.py` |
| 当前 mock 模型 | `jphrl/models/base.py` |

有两个边界必须先说清：

1. 本项目当前的 `MockStructuredModel` 是脚本策略。它明确返回 `token_metadata_status="not_applicable"`，并把 token ID、log-prob、loss mask 留空。因此，它能验证事件和 reward 数据面，不能直接形成 PPO 样本。
2. 本项目已经有 `JointVersion`、可训练的 policy failure、不可训练的 invalid failure 和 token contract。invalid failure 又细分为基础设施异常与 trace contract 异常。目前 `jphrl` 中还没有 AReaL DataProxy adapter。下文给出的是经过代码对照的映射目标，不是宣称这条分布式链已经接通。

## 1. 先分清五个对象

很多困惑来自把 task、session、LLM call、tool observation 和 trajectory 都称为 `轨迹`。它们不是同一个对象。

### 1.1 task 是题目实例

`task_id: str` 用来标识一个待解决的问题。在 calculator smoke 中：

```text
task_id = "add-17-25"
question = "计算 17 与 25 的和。必须先调用 calculator，再根据工具结果回答。"
expected_answer = "42"
```

同一个 task 可以采样多次。也就是说，一个 `task_id` 可以对应多个 session。

### 1.2 session 是一次独立采样

AReaL `SessionStore.start_session(task_id)` 会为这个 task 找一个未使用的整数后缀，形成形如 `add-17-25-0` 的 `session_id`，同时生成一个不透明的 `session_api_key`。

这里的含义是：

| 字段 | 作用 | 是否进入 PPO 特征 |
|---|---|---|
| `session_id` | 关联一次 rollout 中的所有 LLM 调用 | 否 |
| `session_api_key` | 把请求路由到正确 session，并用于鉴权 | 否 |
| `group_id` | 关联同一 task 的一组并行采样 | 否 |

`session_api_key` 是凭据，不是学习信号。它不进入 PPO 张量，也不应写入长期训练产物或日志。

### 1.3 LLM call 是一次模型采样

一次 Agent 任务可以调用模型多次。每次请求和返回在 AReaL 缓存中形成一个 `InteractionWithTokenLogpReward`，并由 `interaction_id` 标识。

对 calculator 成功样本，至少有两次 LLM call：

1. 模型生成工具调用 JSON。
2. 模型读到 calculator 返回的 `42` 后，生成最终答案 JSON。

本项目用 `model_call_id` 关联 `model_request`、`model_response`、`parse_result` 和最终 reward 的归因目标。AReaL 使用 OpenAI completion 或 response 的 ID 作为 `interaction_id`。未来 adapter 必须保存二者的一一映射，不能假定它们天然相等。

### 1.4 tool observation 是环境返回，不是模型动作

第一轮模型输出：

```json
{"tool":"calculator","expression":"17 + 25"}
```

这是模型动作。它由 policy 采样，所以对应的输出 token 应当 `loss_mask=1`。

calculator 返回：

```text
42
```

这是 tool observation。工具本身没有经过语言模型采样，所以不能给这两个字符凭空补一个 old log-prob。它只有在第二次 LLM 请求中被编码为输入 token 时，才进入 `input_ids`，且对应 `loss_mask=0`。

这个区别非常重要：

```text
模型写出工具调用       -> policy action，参与 policy loss
工具执行并返回结果     -> environment observation，不参与 policy loss
模型读取工具结果再回答 -> 新的 policy action，参与 policy loss
```

Hermes 会在 Agent 内部执行工具。AReaL 的代理捕获的是 Hermes 发出的每一次上游 LLM 请求。工具 observation 若被 Hermes 放入下一次请求上下文，就以 prompt token 的身份被捕获。它不需要成为一个单独的梯度样本。

### 1.5 trajectory 是 reward 已经闭合的一段 session 数据

AReaL 在线 session 可以持续存在。一次 `set_reward` 会给某个 interaction 赋分，并在满足结束条件后形成一条 ready trajectory。`trajectory_id` 标识 session 中某一段已经闭合、可以导出的数据。

因此，一条可训练 trajectory 至少满足：

1. 能定位 session。
2. 能列出完整 interaction。
3. 最后一次有效 interaction 有明确 reward。
4. 每个可训练 interaction 都有 token、log-prob 和版本。
5. 数据通过安全与一致性检查。

## 2. calculator 成功样本的逐阶段数据表

下面把 `add-17-25` 从任务开始一直追到权重同步。表中的 `add-17-25-0` 是根据 AReaL session 命名规则写出的教学实例。真实运行若已有同名 session，后缀会递增。`interaction_id` 也用占位名表示，真实值由 API 返回。版本 7 和版本 8 同样是后文沿用的教学数值。

| 阶段 | 发生的事件 | 主要数据 | PPO 如何看待 |
|---:|---|---|---|
| 1 | 读取 task | `task_id=add-17-25`，目标答案 `42` | 还没有训练数据 |
| 2 | 启动 session | `session_id=add-17-25-0`，session key 只用于路由和鉴权，不进入 PPO 张量 | 建立本次采样边界 |
| 3 | Harness 决策 | 当前 smoke 选择 `DIRECT` | 写入 sidecar，原生 AReaL policy tensor 不包含它 |
| 4 | LLM call 1 请求 | system prompt 加 user question | 编码为 prompt token，mask 为 0 |
| 5 | LLM call 1 返回 | `{"tool":"calculator","expression":"17 + 25"}` | 模型输出 token，mask 为 1，保存 behavior log-prob 与 policy version |
| 6 | parser 检查 | 确认是唯一合法 JSON 对象，工具名和参数类型正确 | parser 结果写事件日志，不直接进入 policy tensor |
| 7 | tool 执行 | 安全 calculator 计算 `17 + 25`，返回 `42` | observation 本身没有 policy log-prob |
| 8 | Harness 决策与 verifier | 当前 smoke 选择 `VERIFY`，隐藏校验通过 | 写入 sidecar，决定是否继续 |
| 9 | LLM call 2 请求 | 上下文加入第一轮 action 和 `calculator 返回：42` | 全部作为输入 token，mask 为 0 |
| 10 | LLM call 2 返回 | `{"answer":"42"}` | 模型输出 token，mask 为 1，保存 behavior log-prob 与版本 |
| 11 | 赋 reward | exact evaluator 得到 `reward=1.0` | reward 先赋给最后 interaction |
| 12 | 导出 | Hermes 配置使用 `individual`，`turn_discount=1.0` | reward 从后向前传给该 trajectory 内较早的 interaction |
| 13 | 张量化与 PPO | 形成六个核心张量，计算 advantage 和 actor loss | 只在 mask 为 1 的 token 上反向传播 |
| 14 | 新版本与同步 | 训练器把版本从 7 推进到 8，并同步到推理引擎 | 后续 rollout 使用新权重，旧样本仍记录版本 7 |

本项目成功路径的冻结事件序列有 13 个事件：

```text
episode_started
harness_decision
model_request
model_response
parse_result
tool_result
harness_decision
verifier_result
model_request
model_response
parse_result
reward_assigned
episode_ended
```

注意，13 个事件不等于 13 个 PPO 样本。这个成功例只有两次 LLM call。采用 Hermes v2.0.0 的 `export_style: individual` 时，通常导出两个 interaction 张量项。

## 3. 从 ModelResponse 到六个 PPO 字段

AReaL v2.0.0 的推理返回对象 `ModelResponse` 保存四组关键列表：

```text
input_tokens: list[int]
output_tokens: list[int]
output_logprobs: list[float]
output_versions: list[int]
```

其中 `int` 是 token ID 或整数版本号，`float` 是自然对数概率。若某输出 token 的概率为 `p`，它的 log-prob 定义为：

```text
logprob = ln(p)
```

因为 `0 < p <= 1`，合法 log-prob 必须小于等于 0。

`InteractionWithTokenLogpReward.to_tensor_dict()` 再把一次 interaction 转成六个核心训练字段：

| 字段 | 类型和形状 | 直接定义 | calculator 中的来源 |
|---|---|---|---|
| `input_ids` | 整数张量，形如 `[1, L]` | `input_tokens + output_tokens` | 当前调用的完整 prompt 与模型 completion |
| `loss_mask` | 整数张量，形如 `[1, L]` | prompt 位置为 0，completion 位置为 1 | 只训练工具调用 JSON 或最终答案 JSON |
| `logprobs` | 浮点张量，形如 `[1, L]` | prompt 位置补 0，completion 位置保存采样 log-prob | 第一轮和第二轮各自的 behavior log-prob |
| `versions` | 整数张量，形如 `[1, L]` | prompt 位置填 `-1`，completion 位置保存采样 policy 版本 | 判断样本由哪一版 policy 生成 |
| `attention_mask` | 布尔张量，形如 `[1, L]` | 真实 token 位置为真，padding 位置为假 | 告诉模型哪些位置是有效序列 |
| `rewards` | 浮点张量，形如 `[1]` | 当前 interaction 的标量 reward | 成功为 1.0，policy failure 为 0.0 |

这里 `L` 是这次调用的 prompt token 数与 completion token 数之和。

### 3.1 AReaL 字段与本项目 trace 字段怎样对上

两个仓库的字段命名并不完全相同。不能只看名字相似就直接拼接。

| AReaL v2.0.0 | 本项目 `model_response` | 映射规则 |
|---|---|---|
| `input_tokens` | `input_token_ids` | 一一对应，必须非空才能训练真实 policy |
| `output_tokens` | `output_token_ids` | 一一对应 |
| `output_logprobs` | `output_token_logprobs` | 一一对应，长度必须等于 output token 数 |
| `output_versions` | 当前 trace 没有对应的 token 级整数列表 | `JointVersion.policy` 是字符串 release ID，adapter 还必须原样保存 AReaL 返回的整数 engine version，二者不能互相替代 |
| 完整 `loss_mask` | `completion_loss_mask` | 本项目字段只描述 completion，AReaL 会在前面补 prompt 的 0 |
| `rewards` | `EpisodeTrace.reward` | 只有通过 validity gate 后才能写入 AReaL reward |

本项目的 `JointVersion` 还固定了 Harness controller、Harness artifact、tool schema、parser、environment、evaluator、tokenizer 和 context builder。AReaL 原生 `versions` 张量只表达 policy 生成版本，不能承载整个 `JointVersion`。因此联合版本必须保存在可通过 `episode_id`、`model_call_id`、`session_id` 和 `interaction_id` 关联的 sidecar 中。

### 3.2 一组明确标注的教学 token 例子

下面的 token ID 和 log-prob 是为了说明字段变换而人工选择的数字，不是 Qwen tokenizer 的真实编码，也不是当前 mock smoke 的产物。

假设第一轮 LLM call 得到：

```text
input_tokens       = [101, 102, 103, 104]
output_tokens      = [201, 202, 203]
output_logprobs    = [-0.20, -1.10, -0.35]
output_versions    = [7, 7, 7]
reward             = 1.0
```

那么 `to_tensor_dict()` 的结果是：

| 字段 | 数值 |
|---|---|
| `input_ids` | `[101, 102, 103, 104, 201, 202, 203]` |
| `loss_mask` | `[0, 0, 0, 0, 1, 1, 1]` |
| `logprobs` | `[0.0, 0.0, 0.0, 0.0, -0.20, -1.10, -0.35]` |
| `versions` | `[-1, -1, -1, -1, 7, 7, 7]` |
| `attention_mask` | `[true, true, true, true, true, true, true]` |
| `rewards` | `[1.0]` |

例如第二个输出 token 的采样概率约为：

```text
exp(-1.10) = 0.333
```

`logprobs` 的 prompt 位置填 0，不是说 prompt token 的概率为 1。这里只是占位，因为这些位置的 `loss_mask=0`，不会进入 policy loss。

### 3.3 为什么 trainer 内部还会把 mask 和 log-prob 左移一格

语言模型在序列位置 `t` 的输出 logits 用来预测位置 `t+1` 的 token。AReaL 在 `compute_advantages()` 中把 `loss_mask` 和 behavior `logprobs` 左移一格，使预测动作 token 的位置与该动作 token 的 reward、advantage、log-prob 对齐。

这是因果语言模型的索引对齐，不是修改了哪些 token 属于模型输出。排查数据时应先看导出前的直观 mask，再看 trainer 内部左移后的计算位置，不能把两者混为一谈。

## 4. reward 怎样从任务结果回到两次 LLM call

Hermes v2.0.0 配置使用：

```yaml
export_style: individual
turn_discount: 1.0
```

通常只需给最后一次 interaction 设置任务 reward。AReaL 的 `InteractionCache.apply_reward_discount()` 会按 interaction 创建顺序倒序遍历，使用：

```text
current_reward = current_reward * turn_discount + interaction.reward
```

对两次 LLM call，若只有第二次显式得到 `1.0`，第一次未赋分，并且 `turn_discount=1.0`，则：

| interaction | 显式 reward | 导出 reward |
|---|---:|---:|
| call 2，最终答案 | 1.0 | 1.0 |
| call 1，工具调用 | 未设置，按 0.0 处理 | 1.0 |

若 `turn_discount=0.9`，call 1 会得到 `0.9`。这是一种按 LLM 调用次数做的回传，不是按 token 数做的折扣。

AReaL v2.0.0 的 `to_tensor_dict()` 会把 `reward=None` 转成 `0.0`。因此，安全边界不能等张量化以后再做。基础设施故障若带着 `None` 进入这里，会被错误地伪装成 policy 的零分样本。本项目把 `infrastructure_invalid` 与 `trace_contract_invalid` 固定为 `reward=None`，adapter 必须在调用 AReaL export 或进入训练队列之前丢弃它们。

## 5. rollout 后，PPO 实际计算了什么

先定义三个 policy。对某个 completion token `a_t`：

| 符号 | 定义 | 在 Hermes v2.0.0 配置中的来源 |
|---|---|---|
| `pi_behave(a_t | x_t)` | rollout 时真正采样出 token 的 policy 概率 | 推理引擎返回的 `output_logprobs` |
| `pi_prox(a_t | x_t)` | 训练批次开始时，用 actor 重算得到的近端 policy 概率 | `recompute_logprob: true` 产生的 `prox_logp` |
| `pi_theta(a_t | x_t)` | 当前正在反向传播的 actor 概率 | 训练 forward 的 `logprobs` |

`x_t` 是产生第 `t` 个动作 token 时已经可见的上下文。它包括 prompt 和之前的 token。

Hermes v2.0.0 同时设置 `recompute_logprob: true` 和 `use_decoupled_loss: true`。因此导出的 behavior old log-prob 不会被重算值覆盖。PPO clipping 使用当前 policy 与近端 policy 的比值：

```text
r_t(theta) = exp(log pi_theta(a_t | x_t) - log pi_prox(a_t | x_t))
```

behavior policy 与近端 policy 的差异则可用于判断 rollout 是否太旧，并参与配置中的 rejection sampling：

```text
w_behave = exp(log pi_prox(a_t | x_t) - log pi_behave(a_t | x_t))
```

### 5.1 old log-prob 与 policy version 的数字例子

继续看某一个输出 token。以下仍是教学数值：

```text
rollout behavior log-prob，policy version 7 = -1.10
训练开始时重算的 proximal log-prob       = -1.05
一次参数更新中的 current log-prob        = -0.90
```

先看 rollout 偏离：

```text
w_behave = exp(-1.05 - (-1.10))
          = exp(0.05)
          = 1.051
```

再看 PPO ratio：

```text
r_t = exp(-0.90 - (-1.05))
    = exp(0.15)
    = 1.162
```

v2.0.0 Hermes 配置的 `eps_clip=0.4`，允许区间是 `[0.6, 1.4]`。`1.162` 在区间内，不触发 clipping。如果 current log-prob 变成 `-0.20`，则：

```text
exp(-0.20 - (-1.05)) = exp(0.85) = 2.340
```

此时 ratio 超过 1.4。对正 advantage token，clipped surrogate 会限制这次更新继续放大该动作概率的收益。

版本号解决的是另一类问题。假设这条 rollout 的 completion token 都记录版本 7，训练后得到版本 8：

```text
样本生成时：versions = [7, 7, 7]
PPO 更新后：new_version = 8
权重同步后：后续 rollout 使用 version 8
历史样本：仍然保留 version 7
```

Hermes v2.0.0 配置把 `max_head_offpolicyness` 设为 2。版本字段让 rollout 侧能够约束训练消费速度与生成版本之间的差距。不要把版本 7 的 `versions` 改写成 8，否则就失去了判断数据陈旧程度的依据。

### 5.2 advantage 把序列 reward 分到 token 上

`A_t` 表示在上下文 `x_t` 中选择 token `a_t`，相对于当前基线多带来了多少回报。它是一个与有效预测位置对齐的浮点数。

AReaL 的 actor 先对标量 reward 做 bias、scaling 和可选 normalization，再把任务 reward 放到轨迹末端，并结合可选 KL reward 与 value 估计反向计算 GAE advantage。没有 critic 时，value 按 0 处理。

标准 token 级 PPO 的 clipped 目标可写成：

```text
L_t(theta) = min(
  r_t(theta) * A_t,
  clip(r_t(theta), 1 - epsilon, 1 + epsilon) * A_t
)
```

训练只累加 `loss_mask=1` 的位置。对 calculator 例子：

| 内容 | 是否有 advantage 与 policy gradient |
|---|---|
| system prompt | 否 |
| 用户问题 | 否 |
| 第一轮模型生成的 tool call JSON | 是 |
| calculator 返回的 `42` | 否 |
| 第二轮模型生成的 answer JSON | 是 |

`reward=1.0` 不等于每个 token 都应该更频繁。真正的更新方向还取决于 advantage、ratio、KL 项、mask 和 normalization。

### 5.3 v2.0.0 Hermes 配置有一个不能照抄的细节

固定 tag 的 `examples/hermes/config.yaml` 同时设置 `n_samples: 1` 与 group 级 `reward_norm`。在 v2.0.0 的 normalization 实现中，单元素组的均值就是该元素本身，中心化后的 task reward 会变成 0。本地 AReaL 当前 `main` 已把 Hermes 的 `reward_norm` 和 `adv_norm` 设为 `null`，并在注释中说明 singleton centering 会擦除信号。

所以复现实验时应把 `忠实运行 v2.0.0 原配置` 和 `构造有效学习实验` 分开：前者用于基线验证，后者不能无审计地照抄这组 normalization 设置。

## 6. PPO 更新后怎样同步到 rollout

AReaL v2.0.0 的 `PPOTrainer.train()` 按以下顺序执行：

1. `actor.prepare_batch()` 获取 rollout batch。
2. 可选地计算 critic value、reference log-prob 和 teacher log-prob。
3. 若配置要求，actor 重算 `prox_logp`。
4. `actor.compute_advantages()` 计算 advantage。
5. `actor.ppo_update()` 更新训练侧参数。
6. 暂停 rollout。
7. 令 `new_version = global_step + 1`。
8. `actor.update_weights(versioned_meta)` 把新权重发到推理侧。
9. actor、critic、rollout 和 eval rollout 都切换到新版本。

v2 training controller 的权重更新实现还会在更新前调用 `pause_generation()`，完成后调用 `continue_generation()`。这避免一次生成在权重传输中途读到不一致参数。

可以把这段理解成一个原子交接：

```text
version 7 继续产生完整 rollout
        -> 暂停开始新生成
        -> 训练侧权重 8 传到推理侧
        -> 推理侧声明 version 8
        -> 恢复生成
```

这里更新的只有 model policy。AReaL Hermes 当前没有更新可学习 Harness controller。我们要做 `policy 与 Harness 真正同时更新`，还需要为 Harness action 保存 old Harness log-prob、action mask、controller version 和 Harness advantage，并给它独立执行更新与版本发布。不能把 policy 权重同步成功当成联合训练已经完成。

## 7. Safety boundary 怎样改变训练数据

安全边界不是只负责阻止危险工具。它还负责回答一个统计问题：这次失败能不能归因给被训练的 policy 或 Harness？

本项目把 episode 分成四类：

| `validity_class` | 例子 | reward | 是否进入 policy PPO |
|---|---|---:|---|
| `valid` | 正确调用 calculator 并回答 `42` | 1.0 | 是 |
| `policy_failure` | 非法 JSON、错误答案、超预算、模型请求了被 calculator 拒绝的表达式 | 0.0 | 是，前提是 token 元数据完整 |
| `infrastructure_invalid` | 模型服务断开、Harness controller 服务异常 | `None` | 否 |
| `trace_contract_invalid` | token 长度不一致、版本混用、伪造 token 元数据 | `None` | 否 |

为什么安全拒绝的动作有时仍应训练？例如模型输出了带指数或函数调用的 calculator 表达式。安全 evaluator 不执行它，只记录 `CALCULATOR_REJECTED`。如果这个错误确实由模型动作造成，且 token 与版本完整，它是有效的零分 policy 样本。这样既没有执行危险动作，又能让 policy 降低再次输出该动作的概率。

反过来，网络断开不能记成 0 分。因为模型可能本来会答对，只是我们没有观察到结果。把基础设施故障标成 policy failure，会把随机机器故障注入梯度。

### 7.1 calculator 当前的可执行边界

本项目 calculator 不是 Python `eval`。它只遍历受限 AST，并允许：

```text
整数常量
括号
一元正号和负号
加、减、乘、除
```

它拒绝浮点数、函数调用、名称、导入、幂、整除、取模等语法，并限制：

| 限制 | 数值 |
|---|---:|
| 表达式长度 | 最多 128 个字符 |
| AST 深度 | 最多 16 层 |
| 单个整数位数 | 最多 32 位十进制数字 |
| 结果分子或分母 | 最多 256 bit |

因此 tool observation 是有界数字字符串。未来接 Web、shell 或文件工具时，tool observation 会变成不可信输入。即使它对应 `loss_mask=0`，也必须先做截断、转义、来源标记和权限过滤，因为它仍会影响下一次 policy action。

### 7.2 进入 AReaL 前的最小 gate

未来 adapter 在提交数据前至少应执行以下判断：

```text
if validity_class not in {valid, policy_failure}:
    discard

if reward is None:
    discard

if token_metadata_status != available:
    discard from PPO

if any model call violates token length, log-prob, mask or version contract:
    discard entire episode

otherwise:
    correlate model_call_id with interaction_id
    set reward
    export to AReaL
    keep JointVersion in sidecar
```

这里 `丢弃整个 episode` 是为了避免只留下成功链的一部分，导致 credit assignment 指向错误动作。AReaL 的 offline workflow 对同一 rollout group 中的 agent 异常也采取整组放弃，而不是把异常样本当成正常零分数据。

## 8. Hermes 在这条链里具体做了什么

Hermes 不是 PPO trainer，也不是 AReaL 的替代品。它是 Agent runtime：维护 Agent 行为，决定何时调用 LLM、如何执行工具、怎样把工具结果放回上下文。

AReaL v2.0.0 的 Hermes 接入链可以按六步读：

1. `start_session.py` 向 inference gateway 请求 session，取得 session ID 与 session key。
2. `hermes_loop.py` 把 inference base URL、model 和 session key 发给 Agent Service。
3. Agent Service 的 DataProxy 把这些路由信息注入 `metadata["areal_inference"]`。
4. `examples/hermes/hermes.py` 为每个 session 建一个进程内 `AIAgent`，并让它的上游 LLM 调用经过 AReaL inference gateway。
5. Hermes 在内部循环中执行工具。每次上游 LLM call 都被 AReaL 按同一 session key 捕获。
6. episode 完成后调用 set reward。AReaL 导出 interaction，PPO trainer 消费张量并同步新 policy 权重。

Hermes adapter 把 DataProxy replay 的 history 当作会话事实来源，并关闭自己的持久 memory 与 context files。这样可以避免 Hermes 私有状态与 AReaL 捕获上下文不一致。

对我们的 calculator 项目，最小可行接法是保留 Hermes 的 session 与工具循环，并实现一个明确的桥：

```text
本项目 task 与 Harness
  -> 使用 AReaL session key 发起每次真实 LLM call
  -> AReaL 捕获 token、behavior log-prob、policy version
  -> 本项目安全 parser、tool、verifier 产生事件与 reward
  -> validity gate
  -> set reward 与 export
  -> AReaL PPO
```

Hermes 代码适合参考 session 路由、每 session Agent 生命周期、history ownership 和工具循环。它不能直接提供可学习 Harness 的 old log-prob 与更新器，这部分仍要由本项目实现。

## 9. 对本项目实现的字段验收表

当我们开始接真实 AReaL rollout 时，每条成功或 policy failure episode 应能回答下面的问题：

| 问题 | 必须存在的证据 |
|---|---|
| 这是哪道题的哪次采样？ | `task_id`、`episode_id`、`session_id` |
| 一共调用模型几次？ | 有序 `model_call_id` 与对应 `interaction_id` |
| 每次模型到底看到了什么？ | effective prompt hash，以及 AReaL `input_tokens` |
| 哪些 token 是模型动作？ | `output_tokens` 与 completion mask |
| 这些动作由哪版 policy 采样？ | token 级 `output_versions` 与 sidecar 的 policy release ID |
| 采样概率是多少？ | 每个 output token 的 behavior log-prob |
| 工具返回了什么？ | `tool_result` 事件、结果 hash、tool schema version |
| reward 能归因给谁？ | `target_model_call_ids` 与 `target_harness_decision_ids` |
| 这次失败能训练吗？ | `validity_class` 与非空 reward |
| 整条轨迹能否复现？ | 完整 `JointVersion` 与 Harness spec hash |

缺少任意一项时，先修数据面，不要急着扩大 GPU 规模。GPU 能跑满只说明算力被使用，不说明梯度对应的是正确动作。

## 10. 检查题

先自己回答，再看后面的参考答案。

### 题 1

第二次 LLM 请求中包含 `calculator 返回：42`。这些 token 的 `loss_mask` 应该是 0 还是 1？为什么？

### 题 2

模型第一轮生成 tool call JSON。这个 JSON 是 tool observation 吗？它的 token 是否参与 policy loss？

### 题 3

某输出 token 在 rollout 时的 log-prob 是 `-0.8`，训练当前 log-prob 是 `-0.6`。若近端 log-prob 就等于 rollout log-prob，PPO ratio 是多少？

### 题 4

模型服务在生成最终答案前断开。应该写 `reward=0.0` 还是 `reward=None`？是否进入 PPO？

### 题 5

本项目 mock 模型成功回答 `42`，但它的 `token_metadata_status="not_applicable"`。这条 episode 能直接进入 AReaL PPO 吗？

### 题 6

AReaL `versions` 已经记录 policy version，为什么还要保留本项目 `JointVersion`？

### 题 7

一次成功 calculator episode 有 13 个 trace event 和两次 LLM call。采用 `export_style: individual` 时，训练项更接近 13 个还是 2 个？

### 题 8

PPO 更新得到 version 8 后，是否应把旧 rollout 的 `versions=[7, 7, 7]` 改成 8？

## 11. 参考答案

1. 是 0。`42` 由工具产生，在第二次调用中只是模型输入。它会影响模型输出，但不是模型采样的动作。
2. 不是。tool call JSON 是模型动作，其 completion token 应为 `loss_mask=1`，参与 policy loss。工具执行后的返回值才是 observation。
3. `exp(-0.6 - (-0.8)) = exp(0.2)`，约为 `1.221`。
4. 写 `reward=None`，归类为 `infrastructure_invalid`，在张量化前丢弃。记 0 会错误惩罚 policy。
5. 不能。当前脚本模型没有真实 token、behavior log-prob 与生成版本。它只能用于数据面 smoke。
6. AReaL token 级 `versions` 只表示 policy 生成版本。`JointVersion` 还固定 Harness、工具、parser、环境、evaluator、tokenizer 与 context builder，决定整条轨迹能否复现和正确归因。
7. 更接近 2 个。trace event 是可审计事件，PPO interaction 来自 LLM call。parser、tool 与 verifier 事件不会各自变成 policy 张量。
8. 不应修改。版本 7 是样本来源事实。版本 8 只用于同步后的新 rollout。

## 12. 本课结束时应能复述的一句话

一条 Agent 轨迹变成 AReaL PPO 数据，需要把每次 LLM call 还原成输入 token、模型输出 token、输出 token 的 behavior log-prob、采样版本、loss mask 与经过安全归因的 reward。整段 JSON 事件日志用于审计和关联，不能直接替代这些张量。PPO 只对模型动作位置计算 advantage 和 loss，更新后以新版本原子同步到推理侧。
