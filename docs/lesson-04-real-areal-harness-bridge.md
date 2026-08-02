# 第四课：一次真实 AReaL rollout 怎样与 Harness 决策对齐

第二课解释了 AReaL 的六个训练字段，第三课解释了一份 episode reward 为什么要拆成 policy 与 Harness 两条 credit 链。现在补上中间最容易被一句话带过的部分：真实 AReaL rollout 和真实 Harness 决策，怎样确定属于同一次交互？Harness 的动作是否真的改变了模型输入？rollout 时保存的 log-prob 能否由同一个推理后端重新算出来？

先给结论。当前 bridge 代码尝试把下面四个对象连在一起：

```text
Harness state 与 decision
        -> Harness instruction 改写 prompt
        -> AReaL 生成 ModelResponse 与六字段 tensor_dict
        -> request_id 关联的 bridge sidecar 与同后端复算记录
```

在本课记录的 v5 GPU run 中，这四类对象都已落盘，但整条审计链没有通过预注册的同后端概率门。因此它是一份真实 interaction 的失败诊断证据，不是一份通过验收的数据集，更不是训练链。它保存了终局 reward，也保存了 policy 与 Harness 两类 credit 的目标 ID，但两个 advantage 都是 `None`。它没有调用 policy optimizer，没有调用 Harness optimizer，也没有发布新联合版本。因此，本课不宣称 policy 与 Harness 已经联合学习。

## 0. 代码基线与本课边界

本课回填的 GPU 复验使用项目提交：

```text
41e00d9a2215d03c1108d9728d0a4a8c20752a7a
```

远端实验固定使用 AReaL v2.0.0 提交：

```text
fee938eada49208a5aabdbc1095730a13076a349
```

主要代码路径如下：

| 作用 | 路径 |
| --- | --- |
| 一次 Harness 决策与真实 AReaL 请求 | `jphrl/areal_joint_bridge_workflow.py` |
| prompt、JointVersion 与 sidecar 契约 | `jphrl/trajectory/areal_joint_bridge.py` |
| AReaL ModelResponse 六字段契约 | `jphrl/trajectory/areal_trace_contract.py` |
| rollout 和同 controller 复算 | `scripts/run_areal_joint_bridge_eval.py` |
| prompt、log-prob、版本与 artifact 审计 | `scripts/verify_areal_joint_bridge.py` |
| SGLang score 尾部兼容与 token-ID 绑定 | `jphrl/compat/sglang_score.py` |
| AReaL SGLang engine 薄适配层 | `jphrl/areal_sglang_compat.py` |
| 固定提交、GPU、目录和凭据的运行入口 | `scripts/run_areal_joint_bridge.sh` |
| 本地联合发布边界 | `jphrl/joint_release.py` |

本课讨论 bridge 的代码契约以及失败 run 中已经落盘并可复核的数据。任何未执行或未通过的检查都会明确标出。下面这些工作不在本课证据范围内：

1. 从 terminal reward 计算 policy advantage。
2. 从 terminal reward 计算 Harness advantage。
3. 执行 AReaL PPO optimizer step。
4. 执行 Harness optimizer step。
5. 把两个候选组件作为一个新 `JointVersion` 发布。
6. 比较训练前后的任务指标。

## 1. 先看一次交互的输入、事件和输出

### 1.1 输入是什么

一次 bridge workflow 接收一个 GSM8K 数据项 `data`。运行环境还提前固定了以下输入：

| 输入 | 类型 | 作用 |
| --- | --- | --- |
| `data` | 一个题目对象 | 提供基础 prompt 和终局 reward 所需答案 |
| `TabularHarnessController` checkpoint | 可恢复的 controller 状态 | 产生一次 Harness 动作及其旧概率 |
| tokenizer snapshot | 固定目录 | 把基础 prompt 与有效 prompt 转成 token ID |
| behavior model snapshot | 固定目录 | 由 SGLang 加载并生成输出 |
| expected policy version | 非负整数 | 本次实验固定为 engine version 0 |
| AReaL 与项目 commit | 40 位 Git 对象 ID | 标记代码来源 |
| dataset revision | 数据快照 commit | 标记环境来源 |

这里的 behavior model 是生成动作的 policy 快照。它不是训练后候选模型，因为当前 workflow 根本没有训练步骤。

### 1.2 一步事件怎样发生

workflow 中一次 interaction 的顺序是：

1. 从 `data` 提取 `base_messages`。
2. 把可观察状态编码为 `HarnessState`。当前状态固定 `turn=0`、`remaining_tool_calls=0`、`remaining_model_retries=0`。
3. 在采样前保存 Harness controller checkpoint。
4. Harness 从五个动作中采样一个动作，产生 `decision_id`、old Harness log-prob、action mask、controller version 和 `harness_loss_mask`。
5. 根据所选动作，在原始消息前加入一条有界 system instruction，得到 `effective_messages`。
6. 分别 tokenize 基础消息和有效消息，得到 `base_input_ids` 与 `effective_input_ids`。
7. 创建 `ModelRequest`。请求的 `rid` 等于本次唯一 `request_id`，输入是 `effective_input_ids`。
8. AReaL 推理引擎生成 `ModelResponse`，GSM8K evaluator 给出 terminal reward。
9. `InteractionWithTokenLogpReward.to_tensor_dict()` 产生六字段张量。
10. workflow 构造并验证 bridge sidecar，然后写入 artifact。
11. controller 收齐所有 interaction 后，在销毁前调用同一个 controller 的 `compute_logp()`。
12. 独立验证器检查 prompt 生效、同后端 log-prob、版本、路径、权限和敏感字段。

这 12 步中没有 `optimizer.step()`。第 8 步得到 reward，第 11 步重算 log-prob，都没有修改模型权重。

### 1.3 输出是什么

每次 interaction 产生两类主 artifact：

```text
bridge record
├── AReaL ModelResponse 与六字段 tensor_dict
├── Harness state、decision 与采样前 checkpoint
├── base prompt 与 effective prompt
├── JointVersion
├── terminal reward 与两个 credit target
└── record_sha256

same-backend score record
├── request_id 与 bridge_record_sha256
├── trajectory_binding_sha256
├── stored_logprobs 与 rescored_logprobs
├── controller API/lifecycle、backend 与 engine version
├── tail parser、精确 token-ID 绑定与 RTensor localization
└── record_sha256
```

bridge record 回答这次交互发生了什么。score record 回答同一个后端能否在同一条 token 序列上重现 rollout log-prob。两者通过 `request_id`、bridge hash 和 trajectory hash 连接。

## 2. AReaL 六字段在真实 bridge 中是什么

设本次 interaction 的 prompt 有 `P` 个 token，模型输出有 `G` 个 token，总长度为：

```text
L = P + G
```

当前实验每个 workflow 只采一个 response，所以 batch size `B=1`。六个字段如下：

| 字段 | 当前形状 | 直接定义 |
| --- | --- | --- |
| `input_ids` | `[1, L]` | prompt token 后接模型输出 token |
| `loss_mask` | `[1, L]` | 前 `P` 个位置为 0，后 `G` 个位置为 1 |
| `logprobs` | `[1, L]` | prompt 位置填 0，输出位置保存 rollout old log-prob |
| `versions` | `[1, L]` | prompt 位置填 `-1`，输出位置保存 engine version |
| `attention_mask` | `[1, L]` | 当前无 padding 的有效位置全部为真 |
| `rewards` | `[1]` | 当前 interaction 的 terminal reward |

六字段不是六份彼此独立的数据。它们在同一个序列位置上对齐。对位置 `i`：

```text
input_ids[0][i]       是这个位置的 token ID
loss_mask[0][i]       决定这个位置是不是模型动作
logprobs[0][i]        是 rollout 时该动作的旧 log-prob
versions[0][i]        是生成该动作的 engine version
attention_mask[0][i]  决定这个位置是不是有效 token
```

如果 `loss_mask[0][i]=0`，这个位置是 prompt。它的 `logprobs=0` 和 `versions=-1` 都是占位，不表示概率为 1，也不表示模型版本是负数。

### 2.1 ModelResponse 与 tensor_dict 必须逐项往返一致

AReaL `ModelResponse` 先提供：

```text
input_tokens
output_tokens
output_logprobs
output_versions
stop_reason
```

bridge validator 再要求：

```text
input_ids = input_tokens + output_tokens
loss_mask = [0] * len(input_tokens) + [1] * len(output_tokens)
logprobs 的输出后缀 = output_logprobs
versions = [-1] * len(input_tokens) + output_versions
```

它还要求输出 token、输出 log-prob 和输出 version 的长度完全相等，所有 log-prob 都是有限非正数，单轮输出只能有一个 policy version。任一条件不满足，整条 bridge record 不能写入。

### 2.2 为什么 trajectory hash 要覆盖六字段

score schema v5 对以下六个字段计算 `trajectory_binding_sha256`：

```text
input_ids
loss_mask
logprobs
versions
attention_mask
rewards
```

这样做不是为了隐藏数据。SHA-256 在这里用于绑定内容：如果有人只替换 log-prob、reward 或 mask，score record 中的 trajectory hash 就不再等于 bridge record 的六字段 hash。

## 3. `request_id` 为什么是这条链的主连接键

`request_id` 是一个非空字符串。本次 workflow 用随机 UUID 的十六进制文本创建它。这个 ID 在一次 interaction 中经过四个位置：

```text
ModelRequest.rid
    = Interaction.interaction_id
    = bridge_record.request_id
    = same_backend_score.request_id
```

它回答的是身份问题：这些记录是否描述同一次请求。

仅有 `request_id` 还不够。一个 ID 可以被错误复用，artifact 内容也可能在复制时错位。因此当前 score schema v5 还检查：

1. `bridge_record_sha256` 必须等于对应 bridge 的完整记录 hash。
2. `trajectory_binding_sha256` 必须等于 bridge 六字段 hash。
3. score 中的 `input_ids`、`loss_mask`、`versions` 和 stored log-prob 必须逐项等于 bridge。
4. 每个 request ID 只能出现一次。
5. 每个 score 文件必须恰好消费一次，不能多一个未使用文件，也不能让两个 bridge 复用同一个 score。

可以把这组连接写成：

```text
request_id             连接一次请求的身份
bridge_record_sha256   连接完整 sidecar 内容
trajectory hash        连接六字段训练数据
```

### 3.1 sidecar 到底是什么

sidecar 是与 AReaL policy tensor 并排保存的附加记录。它不改变 `tensor_dict` 的原生字段，而是补上 AReaL 六字段本身表达不了的信息：

| sidecar 部分 | 保存什么 |
| --- | --- |
| `harness.state` | Harness 在决策前看见的状态 |
| `harness.decision` | 动作、old log-prob、mask、logits、controller version |
| `controller_checkpoint_before_decision` | 重放下一次采样所需的 RNG 和参数状态 |
| `prompt_binding` | 基础消息、有效消息、两组 token 和各自 hash |
| `policy_binding` | policy release ID 与预期 engine version |
| `credit_binding` | terminal reward、两类 target ID、尚为空的 advantage |
| `joint_version` | policy、Harness、数据、tokenizer、evaluator 等联合身份 |

sidecar 不是完整的 AReaL PPO batch，也不是 Harness update batch。它当前解决的是对齐和来源问题。

## 4. 怎样证明 Harness prompt 真的生效

只把 Harness decision 写进 JSON，不能证明模型看到了它。当前 bridge 要通过四层检查。

### 4.1 第一层：动作决定一条确定的 instruction

五个 Harness 动作各自映射到一条固定 instruction。`inject_harness_instruction()` 在基础消息前加入：

```text
role = system
content = "JPH Harness action=<ACTION>. <固定 instruction>"
```

输入消息必须是对象列表，每条消息必须有非空 `role` 和字符串 `content`。动作不能注入任意自由文本，它只能选择当前 artifact 中已经固定的五条 instruction 之一。

### 4.2 第二层：基础 prompt 与有效 prompt 必须不同

workflow 分别计算：

```text
base_input_tokens      = tokenize(base_messages)
effective_input_tokens = tokenize(effective_messages)
```

validator 要求两组 token 非空、全部是非负整数，并且：

```text
base_input_tokens != effective_input_tokens
prompt_tokens_changed is True
```

因此，只有 sidecar 字段变化而 token 不变的伪效果会被拒绝。

### 4.3 第三层：AReaL 必须消费 effective tokens

创建 `ModelRequest` 时，`input_ids` 直接取 `effective_input_ids`。生成后，validator 再检查：

```text
ModelResponse.input_tokens == effective_input_tokens
```

这一步把 Harness 决策从 prompt 记录连接到 AReaL 实际生成输入。

### 4.4 第四层：独立重新 tokenize

artifact 验证器重新加载固定 tokenizer snapshot，分别对 `base_messages` 和 `effective_messages` 调用同一 chat template，然后检查：

```text
重新计算的 base tokens      = sidecar base tokens
重新计算的 effective tokens = sidecar effective tokens
重新计算的 effective tokens = ModelResponse.input_tokens
```

这比相信 workflow 自己写下的 token 列表多了一次独立计算。

### 4.5 Harness decision 本身也能重放

sidecar 保存的是采样前 controller checkpoint。validator 从 checkpoint 恢复 `TabularHarnessController`，在同一个 `HarnessState` 上再次采样，并逐项比较：

```text
action
old_harness_logprob
controller_version
action_ids
action_mask
pre_mask_logits
harness_loss_mask
```

这一步证明记录的 Harness 动作与 checkpoint、状态和 RNG 一致。它仍然不代表 Harness 已经更新。更新需要非空 Harness advantage 和一次 optimizer 调用，当前 bridge 两者都没有。

## 5. 同一个 controller 为什么还要重算 log-prob

rollout 已经保存 `stored_logprobs`。重算的目标是检查：给定同一 token 序列、同一后端和同一权重版本，推理服务能否再次给出足够接近的 log-prob。

### 5.1 复算发生在 controller 生命周期的哪个位置

执行顺序是：

```text
controller.initialize()
    -> submit rollout
    -> controller.wait()
    -> RTensor.localize(results)
    -> version_before = controller.get_version()
    -> controller.compute_logp(results)
    -> RTensor.localize(rescored_logprobs)
    -> version_after = controller.get_version()
    -> 写 same-backend score record
    -> controller.destroy()
```

`compute_logp()` 在 `wait()` 之后、`destroy()` 之前运行。它没有新建第二个推理服务，也没有加载另一个模型目录。运行器还要求：

```text
version_before = version_after = expected_policy_version
```

版本发生变化时，运行立即失败。

两次 `RTensor.localize()` 只把 AReaL controller 返回的远端张量引用取回当前进程：第一次让 rollout 六字段可被同一 controller 重打分，第二次让重算结果可被严格序列化和审计。它们不修改 token、mask、log-prob 或模型权重。score provenance 固定记录 controller API v1、同 controller 生命周期、尾部先截取再转换、token-ID 精确绑定以及这两次传输本地化；任一 provenance 字段不同都拒绝。

### 5.2 固定 AReaL 返回什么形状

固定 AReaL 提交中的 `RemoteInfEngine.compute_logp()` 对每条 trajectory 创建：

```text
out = zeros_like(loss_mask, dtype=float32)
```

所以返回值是一组与 `loss_mask` 同形的 `[B, L]` 张量。它只在 `loss_mask` 非零位置写入后端评分，prompt 位置保持 0。当前 `B=1`，验证器因此按单行 `[1,L]` 检查。

SGLang score API 返回目标后缀 token 的 log-prob。bridge 的 trace contract 已经要求 `loss_mask` 是 prompt 的连续 0 后接输出的连续 1，因此后缀评分与 trainable token 位置能够一一对应。

这里的 SGLang regular request 是：把完整 `input_ids` 作为已经给定的 token 序列，请服务端返回其中目标后缀各 token 的条件 log-prob，同时令 `max_new_tokens=1` 以走普通 `/generate` 接口；这一步不是继续采样一段新答案。固定 AReaL v2.0.0 的 parser 会先把整个 `input_token_logprobs` 数组转换为浮点数，但当前 SGLang response 可能在目标后缀前返回一个 `None` 占位。这不是目标 token 的分数。项目的薄适配层先截取最终 `target_len` 项，再转换 log-prob；后缀中的 `None`、非有限值、正 log-prob 或非法 token ID 仍然立即拒绝。

schema v5 又增加了一个更强的条件。构造 score request 时先保存：

```text
expected_token_ids = input_ids[-target_len:]
```

解析 response 时，适配层要求 SGLang 返回的最终 `target_len` 个 token ID 与这组 ID 在长度、顺序和值上逐项相等，然后才接受对应 log-prob。单个 worker 的 score 路径是同步串行的 `build -> HTTP -> parse`；解析成功或失败后都会清空本次绑定，传输在 parse 前失败时，下一次同步 build 会覆盖失效绑定。这样既不会让一次网络故障永久污染 worker，也不会接受没有当前请求绑定的 response。

### 5.3 门禁具体计算什么

对每个 `loss_mask=1` 的位置 `t`，定义：

```text
delta_t = rescored_logprob_t - stored_logprob_t
ratio_error_t = abs(exp(delta_t) - 1)
```

`exp(delta_t)` 是同一 token 的复算概率与 rollout 概率之比。当前预先写入代码的门禁是：

```text
mean(ratio_error_t) <= 0.02
max(ratio_error_t)  <= 0.10
```

也就是平均概率比误差不超过 2%，任一 token 不超过 10%。这两个阈值在运行前已经写进代码；本课记录的 v5 run 实测只有 `1/4` 条轨迹同时通过，具体数值见第 10 节。它们没有因失败而被放宽。

## 6. 为什么 frozen HF 复算只作跨后端诊断

同 controller 复算和 frozen Hugging Face 复算回答的是两个不同问题。

| 复算方式 | 输入 | 计算实现 | 回答的问题 |
| --- | --- | --- | --- |
| same-controller | AReaL 返回的完整 trajectory | 原 SGLang controller 的 `compute_logp()` | rollout 自己的概率记录能否由同一服务重现 |
| frozen HF | 同一模型 snapshot 与 token 序列 | `AutoModelForCausalLM` 直接前向 | 换一个推理实现后，log-prob 相差多少 |

即使两边加载同一模型 snapshot，SGLang 与 Hugging Face 仍是两个计算后端。它们的批处理、数值精度、算子实现和 log-prob 提取路径不完全相同。跨后端差异不能自动解释成数据损坏，也不能拿来替代同后端一致性检查。

因此，bridge verifier 对 frozen HF 使用：

```text
require_all_passed = False
```

这句话的准确含义是：HF 报告中的数值阈值不决定 bridge 是否通过。它不表示 HF 计算可以不执行。如果模型加载失败、没有产生报告或报告无法序列化，整个 verifier 仍会失败。

同后端 2%/10% 门禁才是 policy log-prob 的硬门禁。HF 报告保留下来，用于观察跨后端偏差、排查 token shift 或模型 snapshot 错配，但不把两个后端强行当成位级相同的实现。

## 7. `JointVersion` 固定了哪些共同条件

`JointVersion` 是一个不可变记录。它的输入是本次交互依赖的各组件版本，输出是一个由这些字段计算出的 `version_id`。

当前 bridge 填写：

| 字段 | 当前含义 |
| --- | --- |
| `policy` | behavior snapshot revision 加 engine version 0 |
| `harness_controller` | 采样 Harness 动作的 Tabular controller version |
| `harness_artifact` | 五条 instruction 与 context builder 的内容 hash |
| `tool_schema` | `no-tools-single-turn-v1` |
| `parser` | 固定 AReaL GSM8K reward 实现及 AReaL commit |
| `environment` | GSM8K test snapshot revision |
| `evaluator` | 固定 AReaL GSM8K evaluator 及 AReaL commit |
| `tokenizer` | behavior snapshot revision 对应的 HF tokenizer |
| `context_builder` | `gsm8k-harness-prompt-v1` |

`request_id` 与 `JointVersion` 不能互相替代：

```text
request_id    回答是哪一次 interaction
JointVersion  回答这次 interaction 使用哪一组组件
```

bridge verifier 要求所有计划内记录使用同一个 `JointVersion`，同时要求 request ID 和 Harness decision ID 各自唯一。这样才能表达多次独立请求共享同一行为版本，而不是把多次请求误认为同一件事。

## 8. reward 已经存在，为什么 advantage 仍然必须为空

当前 interaction 有一个有限标量：

```text
raw_terminal_reward = interaction.reward
```

它评价整个 GSM8K interaction 的终局结果。sidecar 还提前保存了两个目标：

```text
policy_target_model_call_id = request_id
harness_target_decision_id  = decision_id
```

目标 ID 只说明将来 credit 应该贴到哪类动作上，不说明 credit 数值是多少。当前 record 明确写：

```text
status = raw-terminal-outcome-only
policy_advantage = None
harness_advantage = None
```

validator 反而要求这两个字段必须是 `None`。如果有人直接把 terminal reward 复制成 advantage，bridge 会拒绝这条 record。

原因可以从定义看出来。reward 是 episode 或 interaction 的结果：

```text
R in R
```

advantage 是某个具体动作相对参照值多带来的预期回报：

```text
A_policy(request_id, token_position)
A_harness(decision_id)
```

从一个 `R` 得到这两类 `A`，还需要各自的 baseline、归因规则、mask 和版本检查。当前 bridge 只完成了 target 绑定，没有执行这个计算。

因此，看到真实 reward、真实 token 和真实 Harness old log-prob，仍然不能说联合学习已经发生。

## 9. 四个 safety boundary 分别在哪里

这里先直接定义 safety boundary。它是事件序列中的一个检查点：系统先检查不可信输入和版本条件，只有检查通过，后面的状态改变或外部动作才允许发生。

四个检查点不能合并。生成、工具执行、artifact 写入和联合发布改变的是四种不同状态。

### 9.1 生成前：运行来源与有效 prompt 必须先固定

位置一在 `scripts/run_areal_joint_bridge.sh` 启动 controller 之前。launcher 检查：

1. AReaL HEAD 等于固定 commit，且工作树干净。
2. 项目 HEAD 等于记录的 commit，且工作树干净。
3. 模型和数据 snapshot 存在，并位于 `JPH_ROOT` 内。
4. GPU 未被本项目另一个进程锁住，显存还有预设余量。
5. 管理 API key 为本次随机生成的临时值，不使用默认 key。

位置二在 `jphrl/areal_joint_bridge_workflow.py` 创建 `ModelRequest` 之前。代码先构造 Harness state、保存 checkpoint、采样动作、验证消息结构、注入固定 instruction，再 tokenize `effective_messages`。`ModelRequest.input_ids` 只取这组有效 token。

这里要保持一个准确限制：完整 `JointVersion` record 在 response 返回后才构造并验证。生成前已经通过 launcher 固定了 commit、snapshot 和 expected engine version，但完整 sidecar 一致性属于生成后的 artifact gate，不应说成生成前已经验证了整条 record。

### 9.2 工具前：当前 bridge 没有工具执行路径

当前 Harness state 写明：

```text
remaining_tool_calls = 0
```

`JointVersion.tool_schema` 写明：

```text
no-tools-single-turn-v1
```

所以本次真实 bridge 在工具边界采用的是按设计拒绝：没有 tool request parser，也没有 tool executor。它不能证明 calculator、Web、shell 或文件工具的安全执行。

项目的 calculator smoke 另有一条工具边界。模型输出先经过 JSON 解析、工具名检查和参数类型检查，然后表达式才进入受限 AST evaluator。该 evaluator 只允许整数、括号、正负号和四则运算，并限制长度、深度和结果大小。这个边界属于 calculator runner，不属于当前 GSM8K no-tools bridge。

将来接工具时，检查点必须位于：

```text
模型生成 tool action
    -> parser 与 schema validation
    -> 权限、预算和参数 safety gate
    -> 只有通过后才调用工具
```

不能先执行工具，再靠 artifact validator 补救。

### 9.3 写 artifact 前：先验证内容，再允许落盘

bridge record 有两次内容检查：

1. builder 计算 `record_sha256` 后立即调用完整 validator。
2. writer 在打开文件前再次调用完整 validator。

writer 随后检查：

```text
allowed_root 必须存在
trace_dir 必须位于 allowed_root 内
request_id 只能含安全文件名字符
目标路径仍必须位于 allowed_root 内
```

落盘使用 `O_EXCL`，已有同名文件时拒绝覆盖。文件权限是 0600，目录权限是 0700，写完执行 `fsync`。JSON 使用 `allow_nan=False`，所以 NaN 和 Infinity 不能进入正式 artifact。

same-backend score 也使用独占创建、0600 和 `fsync`。最后的 verifier 还会重新检查 record hash、路径、symlink、权限和敏感凭据。如果任一项失败，就不会生成 `ok: true` 的 audit。

### 9.4 发布前：当前 bridge 在这个边界之前停止

当前真实 bridge 没有产生 policy candidate 或 Harness candidate，也没有调用 `JointReleaseStore.publish()`。所以它没有触发一次真实联合发布。

项目已有的本地发布边界位于 `jphrl/joint_release.py`：

1. policy candidate 与 Harness candidate 先分别验证。
2. 两个 candidate 的组件类型和 version 必须等于待发布 `JointVersion`。
3. 文件锁内比较 `expected_active_release_id`，活动版本变化时拒绝并发发布。
4. policy 与 Harness 对象按内容 hash 写入。
5. manifest 引用两个对象，并在活动指针切换前重新读取和验证对象版本。
6. 最后才原子替换 `active.json`。

这个发布边界当前只接到 toy control plane。把真实 AReaL optimizer 和真实 Harness optimizer 接入后，还需要让两个真实 candidate 通过相同边界。现在不能把本地 release store 的存在写成真实 bridge 已经发布了新模型。

## 10. 最终 v5 GPU 复验结果

本次复验使用 GPU 0，通过显式 tmux socket `/mnt/sdb/ljw/chizm/runtime/tmux/jph.sock` 运行。session 为 `jph-areal-joint-bridge-20260802T140733Z`，pane `%21` 自然以 exit code 1 结束。失败产物原样保留，没有修改阈值，也没有启动 optimizer。

<!-- BRIDGE_GPU_AUDIT_START -->

| 项目 | GPU 复验结果 |
| --- | --- |
| run root | `/mnt/sdb/ljw/chizm/artifacts/areal-joint-bridge/20260802T140734Z` |
| audit 状态 | `failed/invalid`；`audit.json` 未生成 |
| project commit | `41e00d9a2215d03c1108d9728d0a4a8c20752a7a`；远端工作树 clean |
| AReaL commit | `fee938eada49208a5aabdbc1095730a13076a349` |
| score schema | `jph.areal-same-backend-logprob.v5`，4/4 score record |
| bridge record 数 | 4/4 accepted；平均 reward `0.25` |
| 唯一 request / Harness decision ID | `4 / 4` |
| JointVersion | 1 个，ID `63176adac373c03a` |
| prompt token 结构事实 | 4/4 base 与 effective 不同；长度均增加 8 token |
| prompt token 独立复算 | 未执行；前置 same-controller 硬门先失败 |
| same-controller 检查 token 数 | 每条 64，共 256；256/256 通过请求尾 token-ID 精确绑定 |
| same-controller mean ratio error | task0–3：`0.0200701 / 0.00971994 / 0.00874515 / 0.0126955` |
| same-controller max ratio error | task0–3：`0.142759 / 0.0961468 / 0.201738 / 0.117168` |
| 2%/10% 门禁结果 | 仅 task1 同时通过，合计 `1/4`；整个 run 失败 |
| frozen HF 跨后端诊断 | 未执行；不是本次失败的证据来源 |
| 失败后独立路径/权限检查 | 15 文件、7 目录；全部根内，目录 0700、文件 0600，违规 0、symlink 0；不是正式 audit 结果 |
| 失败后独立敏感信息检查 | unsafe 字段 0、默认 key 匹配 0、随机 key 前缀匹配 0；不是正式 audit，且事后未知完整随机 secret，不能声称重新完成其精确值扫描 |
| GPU | 启动 1 MiB，峰值 28,958 MiB，结束 1 MiB；无残留进程 |
| optimizer / joint claim | policy=false、Harness=false、joint learning=false |

<!-- BRIDGE_GPU_AUDIT_END -->

四条 v5 score 的 `request_id` 匹配且唯一，bridge、trajectory 与 score record hash 均通过；`input_ids`、`loss_mask`、`versions`、stored log-prob、project commit 与 engine version `0 -> 0` 也全部匹配。v5 又逐项验证了 256 个目标 token ID，因此已经排除“score response 取错请求尾 token”这一解释。误差数值与 v4 完全相同，失败边界收窄为两次不同的概率计算事件：生成路径逐 token 采样并随手保存所选 token 的 log-prob；重打分路径则给定完整 token 序列 `x_1,...,x_L`，在不重新采样的情况下，对每个目标位置 `t` 输出 `log p_theta(x_t | x_1,...,x_{t-1})`。后者通常称为 teacher forcing。两条路径之间的数值可重现性仍未通过门禁，但当前证据还不能判断差异来自批处理、数值精度、缓存、生成评分实现还是请求语义。

验证器按门禁顺序在 same-controller 检查处停止，所以不能把 bridge record 中的 prompt token 变化写成“独立 tokenizer 复算已经通过”，也不能填写不存在的 HF 报告或 `audit.json`。这是有意的 fail-closed 行为。

## 11. 当前 bridge 能证明什么，不能证明什么

### 11.1 代码契约已经明确的事实

1. Harness decision 在 AReaL 生成前变成固定 prompt instruction。
2. bridge 保存 AReaL 六字段、Harness decision 和完整 `JointVersion`。
3. `request_id`、bridge hash 与 trajectory hash 连接 interaction 和 score。
4. 同一 controller 在销毁前复算 log-prob，前后 engine version 必须不变。
5. same-controller 2%/10% 阈值是硬门禁。
6. frozen HF 只提供跨后端差异报告。
7. reward 已绑定两类 target，但两个 advantage 被强制保持 `None`。
8. evidence scope 明确把两个 optimizer update 和 joint learning claim 标为 `False`。

### 11.2 本次 v5 GPU run 已确认和未确认的事实

1. 已确认 4 条真实 rollout 被接收，产生 4 份 bridge 和 4 份 v5 score record。
2. 已确认每条 effective prompt 比 base prompt 多 8 个 token；独立 tokenizer 复算因前置门失败而未执行。
3. 已确认 256/256 score token ID 与请求尾部逐项相等；同后端 mean/max 实测如第 10 节，只有 1/4 轨迹完全通过。
4. HF 跨后端诊断未执行，因此本次 run 没有新的 HF 偏差数字。
5. 失败后的独立只读检查确认 artifact 位于允许根目录内，权限、symlink、默认 key、随机 key 前缀和敏感字段均未发现违规；正式 verifier 没有运行到这些阶段，且事后无法用未知的完整随机 secret 重新做精确值扫描。这不改变概率门失败的判定。

### 11.3 当前没有证据支持的结论

1. policy 参数已经改变。
2. Harness 参数已经改变。
3. policy advantage 或 Harness advantage 已经计算。
4. 新的 policy 与 Harness 已经作为一对发布。
5. 任务正确率因为联合训练而提高。
6. 当前 no-tools bridge 已经证明任意外部工具安全。

## 12. 把整条 bridge 压缩成一张图

```text
固定 commit、snapshot、dataset、engine version
                        |
                        v
GSM8K base_messages -> HarnessState
                        |
                        v
            TabularHarnessController.choose()
                        |
             decision + old logprob
                        |
                        v
          注入固定 Harness instruction
                        |
        base tokens != effective tokens
                        |
                        v
      ModelRequest(rid=request_id, effective tokens)
                        |
                        v
       AReaL ModelResponse + terminal reward
                        |
                        v
      Interaction.to_tensor_dict() 六字段
                        |
          +-------------+-------------+
          |                           |
          v                           v
bridge sidecar                 same controller
Harness + prompt +             compute_logp()
JointVersion + targets         version 不变
          |                           |
          +-------------+-------------+
                        |
                        v
       request_id + 两层 hash + 精确 token-ID tail
                        |
                        v
        预注册 mean<=2%、max<=10% 概率门
                        |
                        v
                本次仅 1/4 通过，停止
                        |
                        v
 prompt/HF 后续检查未执行；没有 optimizer 或 publish
```

## 13. 检查题

### 题 1

为什么 `request_id` 不能替代 `JointVersion`？

### 题 2

sidecar 里已经有 Harness action。还要检查 `ModelResponse.input_tokens == effective_input_tokens` 吗？

### 题 3

`loss_mask` 的 prompt 位置为什么是 0？Harness instruction 明明会影响输出，它是否应该参与 policy loss？

### 题 4

同 controller 复算前后 engine version 都是 0，说明了什么？没有说明什么？

### 题 5

frozen HF 的某个 token 与 SGLang 有数值偏差，能否单凭这件事判定 rollout log-prob 损坏？

### 题 6

terminal reward 为 1.0，能否直接把 `policy_advantage` 和 `harness_advantage` 都写成 1.0？

### 题 7

当前 bridge 的 `remaining_tool_calls=0`。能否据此说项目已经验证 calculator 或 shell 工具安全？

### 题 8

为什么 artifact writer 要在 builder 已经验证过一次以后再验证一次？

### 题 9

当前项目有 `JointReleaseStore.publish()`。为什么本次 bridge 仍不能宣称完成联合发布？

## 14. 参考答案

1. `request_id` 标识一次请求，`JointVersion` 标识该请求依赖的 policy、Harness、tokenizer、数据和 evaluator 等组件版本。多个 request 可以共享一个 `JointVersion`。
2. 要检查。只记录 action 不能证明该 action 改变了模型输入。有效 token 与 AReaL 实际输入相等，才完成因果链中的数据连接。
3. Harness instruction 是模型输入，不是模型采样的 token，所以 prompt 位置 `loss_mask=0`。它通过改变上下文影响输出梯度，但自身不作为 policy action 求梯度。
4. 它说明 controller 记录的 policy version 没有跨复算改变。它不说明任何 optimizer 已执行，也不说明 model 参数发生了变化。
5. 不能。HF 与 SGLang 是不同计算后端。先看 same-controller 硬门禁，再把 HF 数值作为跨后端诊断。
6. 不能。reward 是终局结果，advantage 是贴到具体动作上的相对 credit。当前 bridge 没有 baseline 或 credit estimator，并且 validator 要求两个 advantage 都为 `None`。
7. 不能。当前实验通过取消工具权限避开了工具执行。它只证明 no-tools 边界，没有覆盖外部工具。
8. builder 验证内存中的记录，writer 验证即将写入的输入。两次检查缩短了调用者绕过契约或在两步之间传入错误记录的路径。
9. 因为本次 bridge 没有产生两个真实 optimizer candidate，也没有调用发布函数。release store 只是已经存在的控制面边界。

## 15. 本课结束时应能复述的一句话

本次 v5 run 用 `request_id` 和内容 hash 把 Harness 决策、有效 prompt、AReaL 六字段、terminal reward 与同 controller log-prob 复算记录连接起来，并用 `JointVersion` 固定所有行为组件；256/256 目标 token ID 精确绑定通过。但是预注册的概率可重现性门只有 1/4 轨迹通过，所以完整 audit 失败，真实 joint optimizer 闸门保持关闭。policy 与 Harness advantage 仍为空，没有任何 optimizer update，也没有发布新联合版本。
