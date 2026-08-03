# 第五课：从可信 AReaL rollout 到真正同时更新 policy 与 Harness

第四课停在一个很具体的位置：我们已经能把一次真实 AReaL rollout、一次 Harness 决策、有效 prompt、旧 log-prob 和版本信息放进同一条可审计记录，但失败的概率一致性门说明，这批数据还不能直接送进正式训练。

这一课继续往后走，回答一个更难的问题：假设 rollout 已经可信，怎样把它变成一批冻结训练数据，分别更新模型 policy 和 Harness controller，最后又只让完整的新二元组对外可见？

先给结论。这里的“同时更新”不要求两个 GPU kernel 在同一个微秒修改参数。它要求下面四件事同时成立：

1. policy 与 Harness 的样本来自同一个冻结联合版本。
2. 两类动作各自保留采样时的旧概率、行为版本、mask 和 credit。
3. 两个 optimizer 都只生成候选版本，任何一边更新失败都不能单独上线。
4. 发布点只能从完整的旧二元组切换到完整的新二元组。

写成状态转换就是：

```text
活动版本 (P_k, H_k)
        |
        | 采样、评价、冻结 batch
        v
冻结批次 B_k = (B_k^policy, B_k^Harness)
        |
        +--> policy optimizer  --> 候选 P~_(k+1)
        |
        +--> Harness optimizer --> 候选 H~_(k+1)
        |
        v
联合校验与原子发布
        |
        +--> accept: (P_(k+1), H_(k+1))
        |
        +--> reject: 继续使用 (P_k, H_k)
```

这也是本课最重要的判断标准：只有左、右两条更新链都产生了真实候选，并且通过联合发布门，才能说完成了一次真实的 policy 与 Harness 联合更新。

## 0. 先划清当前证据边界

截至 2026-08-02，本项目已经有五块不同层级的证据。它们不能互相替代。

| 层级 | 已经做到什么 | 仍缺什么 |
| --- | --- | --- |
| 真实 AReaL bridge | 真实 SGLang rollout、真实 token、真实 Harness prompt 决策、same-controller score 与完整版本绑定 | 上一轮正式 v5 概率门只通过 1/4，不能作为训练 batch |
| C0/C1 概率机制筛查 | 固定模型与 Harness，只改变 SGLang 生成 log-prob 公式；结果为 2/4 通过，`mechanism_supported=false` | 该处理没有解锁 optimizer，也没有解锁预注册的 32 条校准与 32 条封存确认 |
| C2 CUDA Graph 筛查 | commit `fdaa879`；配置门通过；C2a/C2b 在同一 GPU0 串行完整重启，runtime invariants 仅 `disable_cuda_graph: false -> true` | 四条输出 token 全部不同，配对估计量不可识别，`mechanism_supported=false`；不能判断修复或恶化 |
| G1 synthetic 控制面 | 两类样本、两路 credit、toy 双更新、联合 checkpoint、lag0、原子发布和故障矩阵 | 输入是 synthetic trace，policy 与 Harness 都不是真实生产 optimizer |
| AReaL/Hermes 上游 | AReaL 能对模型 actor 执行 PPO 并同步新权重；Hermes 能把真实 Agent LLM 调用路由进 AReaL 采集 | 上游 Hermes 没有学习 Harness controller 的第二个 optimizer，也没有联合二元组发布 |

所以，本课主体不是完整联合训练结果报告，更不是在宣布联合更新已经完成。它是一份从可信数据面走到真实联合更新所需的数据契约和执行顺序说明；下面只追加与这条路径直接相关的 C0/C1 和 C2 数据面结果。

## 2026-08-02 实验更新：C0/C1 与 C2 都没有解锁训练

C0/C1 pair 已经结束。四条配对轨迹中只有 `2/4` 通过原概率门，最终判定是：

```text
mechanism_supported = false
```

观测到的关键数值是：

- C1 的 stored log-prob 相对 C0 确实发生了变化，但逐轨迹变化量只在 `1.19e-7` 到 `4.77e-7` 之间。
- 两条失败轨迹的 max error 分别是 `0.12716` 和 `0.11109`，仍高于预注册的 `0.10` 上限。
- 因为没有达到 `4/4` 通过，所以结果不能解锁 optimizer，也不能解锁后续的 32 条校准与 32 条封存确认。

这里要注意，“C1 改了 stored log-prob”与“这个改动解释了原失败”是两个不同命题。前者只要求数值不完全相同；后者要求变化足够大，而且能让所有预注册样本通过原门。现在看到的变化只有约 `1e-7`，而失败轨迹仍有约 `1e-1` 的 max error。两者相差大约六个数量级，因此这组证据不支持“切换到 original log-softmax 就能修复概率不一致”这一机制解释。

### 为什么必须使用 common rescored target

对第 `t` 条配对轨迹，记共同的 C0 rescored log-prob 为 `q_t`，两种生成设置保存的值分别为 `s_{0,t}` 和 `s_{1,t}`。比较器实际关心的是：

```text
C0 error = distance(s_{0,t}, q_t)
C1 error = distance(s_{1,t}, q_t)
```

两边使用同一个 `q_t`，变化量才只来自 stored log-prob 的生成公式。如果让 C0 对自己的 rescored target 评分、C1 再对另一个 target 评分，那么 error 的变化会同时混入“stored 值变了”和“评分目标变了”两件事。即使 C1 看起来更好，我们也无法判断是哪一件事造成的。

common target 不是为了偏向 C0，而是为了固定被比较的坐标系。它让问题保持为一个可检验的单句：只替换生成端的 log-prob 公式，stored 值相对同一重算目标是否得到足以通过原门的改善？这次答案是否定的。

### C2 已完成：单变量成立，但配对估计量不可识别

C2 运行在 commit `fdaa879`。它的配置门通过：C2a 和 C2b 在同一张 GPU0 上串行执行，每个 cell 都完整重启；两边 runtime invariants 相同，唯一 treatment 是：

```text
C2a: disable_cuda_graph = false
C2b: disable_cuda_graph = true
```

这说明实验配置确实遵守了单变量原则。可是，四条样本在 C2a 与 C2b 中生成的 output token 全部不同。最终比较状态是：

```text
generation_equal = false
score_alignment = false
common-target paired metrics = null
mechanism_supported = false
```

各 cell 自己观察到的 stored-vs-rescored 指标如下。`mean/max` 都按原概率门判断，token 数是该 cell 实际生成的输出长度。

| 轨迹 | C2a：CUDA Graph 开启 | C2b：普通 CUDA Graph 关闭 |
| --- | --- | --- |
| 1 | 61 tokens；`0.0174049 / 0.119537`；fail | 48 tokens；`0.0182898 / 0.132508`；fail |
| 2 | 64 tokens；`0.0156684 / 0.218675`；fail | 64 tokens；`0.00920763 / 0.112976`；fail |
| 3 | 54 tokens；`0.0181521 / 0.085154`；pass | 64 tokens；`0.0122906 / 0.126716`；fail |
| 4 | 64 tokens；`0.0214181 / 0.157949`；fail | 64 tokens；`0.0217560 / 0.119170`；fail |

这些是两组各自的 observed metrics，不是逐 token 配对的 treatment effect。第 2、4 条看起来是 C2b 的 max error 较低，第 1、3 条却较高，其中第 3 条还从 A 的 pass 变成了 B 的 fail。不能据此说关闭 CUDA Graph 修复了问题，也不能说它恶化了问题，因为两边已经不是同一串 token。

### “影响生成轨迹”为什么不等于“解释概率偏差”

stored-vs-rescored 偏差问的是：对同一个上下文 `x_t` 和同一个实际 token `a_t`，生成时保存的 log-prob 与随后重算的 log-prob 相差多少。可检验的对象是同一个二元组：

```text
(x_t, a_t)
```

C2 的 treatment 能让采样输出改变，说明普通 CUDA Graph 设置可能影响了生成轨迹。轨迹一变，后续每一步的上下文和被评分 token 也跟着变。此时 C2a 的 error 与 C2b 的 error 分别属于两个不同对象，不能相减成“关闭 CUDA Graph 对同一 token 偏差的影响”。

因此，`generation_equal=false` 是一个真实观察，但它只支持“treatment 能影响本次生成路径”。它不支持“treatment 解释了 stored-vs-rescored 偏差”。后一个命题要求两边对齐到相同 prompt、相同上下文和相同 token；这次 `score_alignment=false`，所以 common-target paired metrics 必须是 `null`，而不是勉强计算一个看似完整的数字。

### 下一课应先定义可识别的 estimand

下一课不应先承诺 C3，而应先明确究竟要估计什么。`estimand` 是实验希望从数据中识别的那个量。这里至少有两个不同问题：

1. 如果要估计 CUDA Graph 设置对“同一动作的 stored-vs-rescored 偏差”的影响，可以考虑确定性 replay：先冻结 prompt 和完整 output token ID，再让两个 runtime 对同一串 token 评分。这样比较对象始终是同一个 `(x_t,a_t)`。
2. 如果要估计 CUDA Graph 设置对“生成分布或轨迹”的影响，就不能再要求逐 token 配对。需要把输出分布、重复运行和统计单位重新定义，不能用四条不同生成结果的 mean/max 直接代替。

选择哪一个问题会决定实验数据、比较器和门槛的定义。在这个选择完成以前，不应给下一轮实验编号，更不应声称 C2 解锁了 32 条校准、32 条封存确认或 optimizer。

## 1. “冻结 batch”到底冻结了什么

先直接定义。

一个冻结联合批次 `FrozenJointBatch` 是一组只读训练输入。它在两个 optimizer 开始计算之前就已经确定，之后不能再新增轨迹、替换 log-prob、改变 mask、重算 credit 或切换行为版本。

可以把目标类型写成：

```text
FrozenJointBatch = {
    parent_joint_version: JointVersion,
    policy_batch: AReaLPolicyBatch,
    harness_batch: HarnessDecisionBatch,
    episode_ids: tuple[str, ...],
    batch_digest: sha256,
}
```

其中：

```text
AReaLPolicyBatch = Sequence[{
    input_ids:       Int64Tensor[1, L_i],
    loss_mask:       BinaryTensor[1, L_i],
    old_logprobs:    FloatTensor[1, L_i],
    versions:        Int64Tensor[1, L_i],
    attention_mask:  BoolTensor[1, L_i],
    rewards:         FloatTensor[1],
    policy_credit:   FloatTensor[1, L_i],
}]

HarnessDecisionBatch = Sequence[{
    decision_id:              str,
    state:                    HarnessState,
    action:                   str,
    action_ids:               tuple[str, ...],
    action_mask:              tuple[bool, ...],
    pre_mask_logits:          tuple[float, ...],
    old_harness_logprob:      float,
    harness_loss_mask:        0 | 1,
    harness_behavior_version: str,
    harness_credit:           float,
}]
```

这里的 `Int64Tensor[1, L_i]` 表示元素是 64 位整数、形状为一行 `L_i` 列的张量。不同轨迹的 `L_i` 可以不同，组 batch 时再 padding 或 packing。

“冻结”不是说 Python 对象必须使用 `frozen=True`。它首先是一条训练语义：两个 optimizer 必须读取同一份父版本、同一组 episode 和同一批已经封口的数据。

### 1.1 policy 的六字段来自哪里

当前真实 bridge 直接保存 AReaL 的六个字段：

| 字段 | 输入 | 输出含义 |
| --- | --- | --- |
| `input_ids` | prompt token 后接输出 token | 指明模型实际处理的完整 token 序列 |
| `loss_mask` | 与 `input_ids` 同形状 | 1 表示该位置是可训练 policy 动作，0 表示只用于上下文或对齐 |
| `logprobs` | 与 `input_ids` 同形状 | rollout 时行为 policy 对该 token 的旧 log-prob |
| `versions` | 与 `input_ids` 同形状 | 生成每个位置时推理引擎使用的权重版本 |
| `attention_mask` | 与 `input_ids` 同形状 | 表示哪些位置是有效 token |
| `rewards` | 每条 interaction 一个标量 | 终局结果，不等于已经归因后的 token advantage |

本课把六字段中的 `logprobs` 改名解释为 `old_logprobs`，只是为了强调它的训练角色，落盘字段名仍然是 AReaL 的 `logprobs`。

项目中的直接对应位置是：

- `jphrl/trajectory/areal_trace_contract.py`：校验 `ModelResponse` 与六字段逐位置一致。
- `jphrl/areal_joint_bridge_workflow.py`：把 Harness 有效 prompt 送进 AReaL，并构造 bridge record。
- `scripts/run_areal_joint_bridge_eval.py`：提交 rollout，在同一个 controller 销毁前调用 `compute_logp()`。
- `jphrl/trajectory/areal_joint_bridge.py`：把六字段、Harness sidecar、runtime contract 和 `JointVersion` 绑定起来。

### 1.2 当前 `JointDecisionBatch` 还不是完整冻结 batch

项目的 `jphrl/trajectory/joint_batch.py` 已定义两类逐动作样本：

```python
PolicyTokenSample(
    episode_id,
    model_call_id,
    output_position,
    token_id,
    old_policy_logprob,
    policy_loss_mask,
    policy_release_id,
    inference_engine_version,
    advantage,
    credit_source,
)
```

```python
HarnessActionSample(
    episode_id,
    decision_id,
    action,
    action_ids,
    action_mask,
    pre_mask_logits,
    old_harness_logprob,
    harness_loss_mask,
    harness_behavior_version,
    advantage,
    credit_source,
)
```

`JointDecisionBatch` 把这两条流放在同一个 `JointVersion` 下，并检查 ID、版本、mask、old log-prob 和 credit。

但它仍是 sidecar。它没有保存完整 prompt、`attention_mask`、padding/packing 结构和 AReaL actor 所需的完整 tensor batch。因此，真实实现不能把 `JointDecisionBatch` 直接传给 `actor.ppo_update()`。正确做法是让 AReaL 原生 policy batch 与 `JointDecisionBatch` 通过稳定的 `episode_id + model_call_id/request_id` 连接，再一起冻结。

### 1.3 冻结时必须成立的五条不变量

对批次中的每条 episode，至少要检查：

1. `trace.joint_version == active_joint_version`。
2. 每个可训练 policy token 的 `policy_release_id` 等于 `JointVersion.policy`。
3. 每个可训练 policy token 的 engine version 属于本轮允许的行为版本。
4. 每个 Harness 决策的 `harness_behavior_version` 等于 `JointVersion.harness_controller`。
5. policy credit target 和 Harness credit target 都恰好命中一次真实动作，不能缺失、重复或交叉。

本项目当前采用最严格的 lag0 准入：

```text
trace.joint_version 必须完全等于当前 active_joint_version
```

这会拒绝跨越发布点、仍由旧二元组产生的轨迹。它会损失一些吞吐，但能先把联合更新的因果边界做清楚。

## 2. 两种 old log-prob 与两种行为版本

### 2.1 policy old log-prob

对 batch 中第 `b` 条轨迹、第 `i` 个 token，定义：

\[
\ell^{\pi,\mathrm{old}}_{b,i}
=
\log \pi_{P_k}
\left(a_{b,i}\mid x_{b,<i}\right).
\]

各符号的直接含义是：

- `b`：轨迹索引。
- `i`：token 位置。
- `x_{b,<i}`：这个 token 之前的有效上下文。
- `a_{b,i}`：rollout 实际采样到的 token ID。
- `P_k`：采样时活动的 policy 版本。
- 输出：一个不大于 0 的有限实数。

版本字段回答另一个问题：这个数究竟由哪份推理权重产生？

```text
policy_release_id       回答发布身份，例如 P_k
inference_engine_version 回答推理服务内部的权重版本，例如 7
```

两者必须同时记录。发布 ID 相同但 engine 实际没完成权重同步，仍可能生成错误数据；engine version 相同但发布中的 tokenizer、Harness 或 evaluator 不同，也不能视为同一个联合行为版本。

### 2.2 Harness old log-prob

Harness 的一次动作不是 token。设第 `d` 个控制位置看见状态 `z_d`，允许动作由 `action_mask` 给出，实际选中动作是 `u_d`。定义：

\[
\ell^{H,\mathrm{old}}_d
=
\log H_{H_k}(u_d\mid z_d,m_d).
\]

这里：

- `H_k` 是采样时活动的 Harness controller 版本。
- `m_d` 是布尔 action mask，禁止动作的概率必须为 0。
- 输出是实际动作在 masked categorical distribution 下的 log-prob。

当前项目不只相信落盘的 `old_harness_logprob`。`JointDecisionBatch.validate()` 会从 `pre_mask_logits` 与 `action_mask` 重新计算 masked log-softmax，再要求结果与记录值在 `1e-9` 绝对误差内一致。

这一步很重要。没有 Harness old log-prob，就不能计算更新前后 Harness 动作的概率比，也不能区分“这个动作由旧 controller 采样”与“训练后临时补写了一个动作标签”。

## 3. 同一份结果为什么要形成两路 credit

一次 episode 可以只有一个终局奖励：

\[
R_e\in\mathbb R.
\]

但 batch 中有两类动作：

```text
policy 动作：每个输出 token
Harness 动作：每次路由、检索、验证、重试、压缩或终止决定
```

所以需要两个映射：

\[
C^{\pi}_e:
\text{model-call-id}\times\text{token-position}
\rightarrow\mathbb R,
\]

\[
C^H_e:
\text{decision-id}
\rightarrow\mathbb R.
\]

它们的输出分别记为：

\[
A^{\pi}_{e,i}=C^{\pi}_e(i),
\qquad
A^H_{e,d}=C^H_e(d).
\]

`A` 表示 advantage，也就是某个具体动作相对参照值多带来的估计回报。它不是 reward 的另一个名字。

早期真实 bridge 只保存：

```text
raw_terminal_reward
policy_target_model_call_id
harness_target_decision_id
```

并明确要求：

```text
policy_advantage = None
harness_advantage = None
```

这说明当时 target 已经绑定，但 credit estimator 还没有接入。项目 G1 中出现的 `synthetic-policy-credit-fixture-v1` 和 `synthetic-harness-credit-fixture-v1` 只是控制面测试输入，不能拿来训练真实 AReaL 模型。

现在 Q/R/S 已补上这段数据面：Q 保存真实 AReaL 六字段样本和每个 model call 的 decision span；R 保存完整 Harness state、五动作 logits/mask、old log-prob 与 behavior version；S 要求它们来自同一 P record、episode 和 lag-zero `JointVersion`，然后分别执行

\[
A^\pi_{e,i}=R_e-b^\pi_i,
\qquad
A^H_{e,d}=R_e-b^H_d.
\]

这里的两份 baseline map、snapshot ID、source 和 estimator version 都在 batch 冻结前明确保存。Policy advantage 只铺到对应 decision span，credit mask 必须与 AReaL loss mask 完全一致；Harness advantage 则只乘自己的 `harness_loss_mask`。S 明确拒绝 synthetic/placeholder source。它完成的是可持久重验的 credit alignment，仍没有执行 Policy 或 Harness optimizer。

### 3.1 两路 credit 必须分开保存 source

目标系统中的一个 credit 至少应当包含：

```text
DecisionCredit = {
    advantage: float,
    source: str,
    estimator_version: str,
    parent_joint_version_id: str,
}
```

`source` 记录计算来源，例如 policy 的 GAE、group-relative estimator，或者 Harness 的 decision-level return decomposition。它不能只写一个模糊的 `reward`。

分开保存 source 有两个作用：

1. 出现异常梯度时，可以判断错误来自哪条归因链。
2. 做干预测试时，可以只替换一侧 credit，验证另一侧候选严格不变。

但 source 字符串不同仍然不是联合更新证据。真正的证据是：只改 policy credit 时只有 policy candidate 改变，只改 Harness credit 时只有 Harness candidate 改变。

## 4. 数值例子一：一条冻结轨迹怎样产生两组概率比

下面的数值只用于演示计算，不是 C0/C1 的实验产物。

### 第一步：冻结 AReaL 六字段

假设有效 prompt 有两个 token，模型输出两个 token：

```text
input_ids      = [[101, 102, 201, 202]]
loss_mask      = [[  0,   0,   1,   1]]
logprobs       = [[0.0, 0.0, -0.80, -1.20]]
versions       = [[ -1,  -1,     7,     7]]
attention_mask = [[True, True, True, True]]
rewards        = [1.0]
```

前两个位置是 prompt。它们的 log-prob `0.0` 与 version `-1` 是占位值，不参加 policy loss。后两个位置由 engine version 7 生成，旧 policy log-prob 分别是 `-0.80` 和 `-1.20`。

### 第二步：冻结 Harness 决策

假设动作集合和采样前 logits 是：

```text
action_ids      = (DIRECT, VERIFY, REPLAN)
action_mask     = (True,   True,   False)
pre_mask_logits = (0.2,    0.8,    9.0)
chosen_action   = VERIFY
```

`REPLAN` 虽然原始 logit 是 9.0，但 mask 为 `False`，归一化时必须完全排除。两个允许动作的分母是：

\[
Z=e^{0.2}+e^{0.8}.
\]

所以 `VERIFY` 的旧概率与旧 log-prob 是：

\[
p^H_{\mathrm{old}}
=\frac{e^{0.8}}{e^{0.2}+e^{0.8}}
\approx0.645656,
\]

\[
\ell^{H,\mathrm{old}}
=\log 0.645656
\approx-0.437488.
\]

如果落盘值不是约 `-0.437488`，这条 Harness 样本应在冻结 batch 前被拒绝。

### 第三步：分别产生 credit

为了演示，假设两个 estimator 使用不同参照值：

\[
A^\pi=R-b^\pi=1.00-0.30=0.70,
\]

\[
A^H=R-b^H=1.00-0.55=0.45.
\]

两个 policy token 都得到 `0.70`，一次 Harness 决策得到 `0.45`。它们来自同一个终局结果，但 target 和参照值不同。

### 第四步：候选参数重新计算动作概率

假设 policy candidate 对两个已采样 token 的新 log-prob 是：

```text
new_policy_logprobs = (-0.72, -1.25)
```

新旧概率比分别是：

\[
r^\pi_0=\exp(-0.72-(-0.80))
=\exp(0.08)\approx1.0833,
\]

\[
r^\pi_1=\exp(-1.25-(-1.20))
=\exp(-0.05)\approx0.9512.
\]

假设 Harness candidate 对原动作 `VERIFY` 的新 log-prob 是 `-0.35`，则：

\[
r^H=\exp(-0.35-(-0.437488))
\approx1.0914.
\]

此时有三个不同的 ratio：两个属于 policy token，一个属于 Harness 决策。不能把 `r^H` 复制到 token loss，也不能把两个 token ratio 的平均值当成 Harness ratio。

### 第五步：分别进入两条 loss

若两边都采用裁剪比例目标，形式可以写成：

\[
L_\pi
=-
\frac{1}{N_\pi}
\sum_i M_i^\pi
\min\left(
r_i^\pi A_i^\pi,
\operatorname{clip}(r_i^\pi,1-\epsilon_\pi,1+\epsilon_\pi)A_i^\pi
\right),
\]

\[
L_H
=-
\frac{1}{N_H}
\sum_d M_d^H
\min\left(
r_d^H A_d^H,
\operatorname{clip}(r_d^H,1-\epsilon_H,1+\epsilon_H)A_d^H
\right).
\]

`M_i^π` 与 `M_d^H` 分别是 policy 和 Harness loss mask。`N_π` 与 `N_H` 分别是两条流中 mask 为 1 的样本数。

目标项目不一定必须让 Harness 使用 PPO。它也可以使用 REINFORCE、advantage-weighted regression 或离线 RL。必须保持的是输入分流、旧概率可验证、credit target 独立和版本绑定，不能为了共用一段代码把两类动作混成同一个张量语义。

## 5. 两个 optimizer 各自更新什么

### 5.1 policy optimizer

policy optimizer 的类型签名可以写成：

```text
PolicyUpdate:
    (PolicyCheckpoint[P_k], FrozenPolicyBatch[B_k^policy])
    -> PolicyCandidate[P~_(k+1)]
```

输入至少包含：

- 模型参数。
- optimizer state，例如 Adam 的一阶矩、二阶矩和 step。
- 学习率调度器状态。
- 与 batch 对齐的 token、mask、old log-prob、advantage 和行为版本。

输出是候选 policy checkpoint。它不是活动版本。

AReaL 已经提供真实模型侧路径。在 `areal/trainer/rl_trainer.py` 中：

```text
actor.prepare_batch(...)
    -> actor.compute_logp(...), 按配置可选
    -> actor.compute_advantages(...)
    -> actor.ppo_update(adv_batch)
    -> actor.step_lr_scheduler()
    -> actor.update_weights(versioned_meta)
    -> rollout.set_version(new_version)
```

Hermes 的 `examples/hermes/train.py` 直接创建 `PPOTrainer` 并调用 `trainer.train()`；`examples/hermes/config.yaml` 给 actor 配置 Adam、PPO clip、是否复算 log-prob 以及权重同步方式。

这证明 AReaL 有可复用的真实 policy optimizer 和权重同步骨架。它不证明本项目的 Harness sidecar 已经接进这个 batch，也不证明第二个 optimizer 已经存在于同一训练事务。

### 5.2 Harness optimizer

Harness optimizer 的目标类型签名是：

```text
HarnessUpdate:
    (HarnessCheckpoint[H_k], FrozenHarnessBatch[B_k^Harness])
    -> HarnessCandidate[H~_(k+1)]
```

输入至少包含：

- Harness controller 参数或可学习外部状态。
- Harness optimizer state。
- 决策前可观察状态。
- 动作 ID、action mask、采样前 logits 和 old Harness log-prob。
- Harness loss mask、Harness advantage 与 behavior controller version。

本项目的 `jphrl/harness/learning.py` 已有一个可执行的 `TabularHarnessController.updated()`。它验证 behavior version、masked logits 和 old log-prob，再做一次 on-behavior REINFORCE 更新，并返回一个不修改旧 controller 的 candidate。

这条路径证明 Harness 动作可以有独立概率、credit 和非零参数更新，但它仍是小型 CPU tabular controller。生产版本至少还需要 batched Torch 模块、正式 optimizer state、分布式 checkpoint、与真实 AReaL batch 的 join，以及发布前回归评价。

### 5.3 “同时”不等于共享一个 optimizer

错误实现通常有两种：

1. 把 Harness 参数塞进 actor optimizer 的同一个参数列表，但没有独立 Harness action、old log-prob 和 credit。这只是多了一组参数，不是可归因的 Harness RL。
2. policy 更新后立即上线，再慢慢计算 Harness candidate。此时新 rollout 会来自 `(P_(k+1), H_k)`，产生训练计划之外的半版本。

正确语义是：

```text
父版本必须相同：parent(P candidate) = parent(H candidate) = (P_k, H_k)
批次必须相同：episodes(P batch) = episodes(H batch)
发布必须成对：active 只能从 (P_k, H_k) 切到 (P_(k+1), H_(k+1))
```

两个 optimizer 可以串行运行，也可以并行运行。只要它们读取同一冻结父版本，且在联合门之前都只是 candidate，就满足这里的“同时”。

## 6. 候选版本与原子联合发布

### 6.1 candidate 是什么

candidate 是已经算完但尚未对 rollout 服务可见的新组件状态。

项目当前用以下类型表达：

```text
CandidateArtifact = {
    component: "policy" | "harness",
    version: str,
    payload: Mapping[str, object],
}
```

真实接入后，policy payload 不应是把大模型权重直接塞进一个 JSON。更实际的做法是让 payload 保存 checkpoint 路径、内容 hash、optimizer state manifest 和必要元数据；模型、数据与大型 checkpoint 继续位于项目仓库之外。

### 6.2 联合 manifest 是什么

`ReleaseManifest` 同时引用两个内容寻址对象：

```text
ReleaseManifest = {
    release_id,
    parent_release_id,
    joint_version,
    policy_object,
    harness_object,
}
```

`joint_version.policy` 必须等于 policy candidate version，`joint_version.harness_controller` 必须等于 Harness candidate version。任一不等就拒绝。

### 6.3 原子发布的可观察条件

设活动版本最初为：

\[
V_k=(P_k,H_k).
\]

两个 optimizer 产生：

\[
\widetilde V_{k+1}
=
(\widetilde P_{k+1},\widetilde H_{k+1}).
\]

对任何读取活动版本的 worker，在发布期间只允许观察：

\[
V_{\mathrm{visible}}
\in
\left\{
(P_k,H_k),
(P_{k+1},H_{k+1})
\right\}.
\]

下面两个组合永远不能成为活动版本：

\[
(P_{k+1},H_k),
\qquad
(P_k,H_{k+1}).
\]

`jphrl/joint_release.py` 的本地实现按以下顺序工作：

1. 校验两个 candidate 的组件类型和 version。
2. 获取发布文件锁。
3. 比较 `expected_active_release_id`，发现并发变化就拒绝。
4. 分别写入按内容 hash 命名的 policy 与 Harness 对象。
5. 写入同时引用两者的 manifest，并重新读取对象做版本校验。
6. 最后用原子替换切换 `active.json`。

G1 已对 policy 对象写后、Harness 对象写后、active 切换前和切换后注入进程退出。这个证据只覆盖本地 POSIX 发布控制面，不等于真实多机权重服务已经采用了同样的 commit point。

## 7. 数值例子二：两个 optimizer 成功，不代表已经发布

这次用两个标量参数演示候选与发布状态。数字同样只用于教学。

### 第一步：读取同一个父版本

```text
active_release = R17
policy version = P7, 参数 theta = 0.40
Harness version = H3, 参数 phi = -0.20
```

冻结批次的 `parent_joint_version` 必须精确等于 `(P7, H3)`。

### 第二步：policy optimizer 产生 candidate

假设 policy loss 对 `theta` 的梯度是：

\[
g_\pi=-0.30.
\]

学习率为 `0.10`，使用最简单的梯度下降演示：

\[
\widetilde\theta
=0.40-0.10\times(-0.30)
=0.43.
\]

产生 `PolicyCandidate(version=P8, theta=0.43, parent=R17)`。

此时活动服务仍必须使用 `P7`。

### 第三步：Harness optimizer 产生 candidate

假设 Harness loss 对 `phi` 的梯度是：

\[
g_H=0.50.
\]

学习率为 `0.05`：

\[
\widetilde\phi
=-0.20-0.05\times0.50
=-0.225.
\]

产生 `HarnessCandidate(version=H4, phi=-0.225, parent=R17)`。

活动服务仍必须使用 `H3`。

### 第四步：构造完整 manifest

```text
candidate manifest:
    parent_release_id = R17
    joint_version     = (P8, H4)
    policy_object     = hash(P8 checkpoint)
    harness_object    = hash(H4 checkpoint)
```

### 第五步：观察两种失败

情况 A，policy 对象写完后进程退出：

```text
磁盘可能有 P8 object
active 仍是 R17 -> (P7, H3)
```

情况 B，两个对象都写完，但独立回归门拒绝 H4：

```text
磁盘可能有 P8 object 与 H4 object
active 仍是 R17 -> (P7, H3)
```

对象存在不等于发布成功。

### 第六步：只有 active commit point 成功才对外切换

```text
active: R17 -> R18
R18 joint_version = (P8, H4)
```

从这一刻开始，新 episode 才能固定 `(P8, H4)` 采样。已经在 R17 下开始的 episode 若采用 lag0，应在进入下一训练 batch 时被视为 stale 并丢弃，而不是把旧数据伪装成新版本数据。

## 8. 本项目、AReaL 与 Hermes 分别提供哪一段

### 8.1 对照表

| 目标步骤 | 本项目对应位置 | AReaL/Hermes 对应位置 | 当前缺口 |
| --- | --- | --- | --- |
| 真实 Agent LLM 调用进入 rollout | `areal_joint_bridge_workflow.py` 的单轮 GSM8K bridge | `examples/hermes/hermes.py` 从 `areal_inference` metadata 取得 session upstream；DataProxy 按 session 采集 | 当前 bridge 还是 no-tools 单轮，不是完整 Hermes 多轮轨迹 |
| policy 六字段 | `areal_trace_contract.py` 与 bridge record | AReaL inference controller 和 trajectory export | C0/C1 未支持原公式机制；C2 输出不对齐，paired metrics 为 `null`，仍未识别 CUDA Graph 对同一 token 偏差的作用 |
| policy advantage 与 PPO | 尚未接入真实 joint batch | `PPOTrainer.train()` 中 `compute_advantages()` 与 `actor.ppo_update()` | 需要把通过审计的 bridge 数据接回 AReaL 原生 batch |
| Harness 动作概率 | `HarnessDecision`、`JointDecisionBatch`、`TabularHarnessController` | Hermes 内部有工具与循环控制，但上游示例不导出这些控制动作的行为概率 | 需要把可学习 Harness action 显式化，不能只保留工具调用摘要 |
| Harness optimizer | tabular REINFORCE 与 G1 toy updater | Hermes 示例没有第二个 optimizer | 需要生产 Torch Harness controller 和 checkpoint |
| 联合发布 | `JointReleaseStore` 与 G1 故障矩阵 | AReaL 会把新 actor 权重同步给 rollout | 需要把 AReaL weight version 与 Harness candidate 纳入同一个 release manifest |

### 8.2 Hermes 最值得参考的部分

Hermes 示例解决了一个实际接线问题：真实 Agent runtime 里的 LLM 调用怎样经过 AReaL inference gateway，并按 session 形成训练轨迹。

关键路径是：

1. `examples/hermes/start_session.py` 创建每个 episode 的 `sk-sess-*` key。
2. `examples/hermes/hermes_loop.py` 把 inference gateway 地址与 session key 发给 Agent Service。
3. `examples/hermes/hermes.py` 优先读取 DataProxy 注入的 `areal_inference` metadata，把 Hermes 的上游模型调用路由到 AReaL。
4. `examples/hermes/set_reward.py` 给该 session 的 interaction 设置标量 reward。
5. `examples/hermes/train.py` 启动 `PPOTrainer`，由 AReaL 更新模型 actor。

这段实现非常适合参考 policy rollout 的真实接入方式。

### 8.3 Hermes 不能直接替代 Harness 学习链

上游 `HermesAgent` 会执行工具、维护每个 session 的 `AIAgent`，也会在 metadata 中汇总工具调用。但当前示例明确不把 Hermes 内部工具执行发成成对的 `tool_call` 与 `tool_result` 训练事件，因为工具由 Hermes 内部执行。

因此它没有直接提供本项目所需的以下字段：

```text
Harness state before decision
action_ids + action_mask
old_harness_logprob
harness_behavior_version
harness_loss_mask
decision-level credit
Harness optimizer state
```

只把 Hermes 接进 AReaL 并运行 `PPOTrainer`，完成的是“在真实 Harness 中训练模型”。它更新的是模型 policy，不会自动把 Hermes 的路由、工具选择、重试或上下文规则变成第二个可学习策略。

要实现真正的 `(Delta policy, Delta Harness) = (1, 1)`，需要在 Hermes/AReaL 的调用边界显式记录 Harness decision，并让它经过第二条 credit 和 optimizer 链。

## 9. 四个 Safety boundary 为什么要放在四个不同位置

先直接定义。Safety boundary 是状态改变之前的检查点。检查不通过，后面的采样、外部执行、持久写入或版本切换不得发生。

四个边界保护的对象不同，所以不能合并成一个训练结束后的总审计。

### 9.1 采样前：先固定“谁在行动”

采样会产生后续训练依赖的行为分布。采样前至少要固定：

```text
JointVersion V_k
模型与 tokenizer snapshot
Harness checkpoint 与 action schema
dataset revision 与样本位置
推理 server args、采样配置、GPU 和依赖版本
```

当前 C0/C1 launcher 使用 clean `env -i` 入口、同一物理 GPU 串行执行，并把完整固定 runtime contract 与唯一 treatment 字段写入 manifest。C2 也通过配置门，在同一 GPU0 上串行完整重启，除 `disable_cuda_graph` 从 `false` 变为 `true` 外 runtime invariants 相同。真实 bridge 还会在 Harness 采样前保存 controller checkpoint。

这个边界阻止的错误包括：

- C0 与 C1 偷偷继承不同环境变量。
- policy 使用新权重，Harness 仍使用旧 checkpoint。
- Harness 动作记录存在，但有效 prompt 由另一套规则构造。
- seed、temperature 或 CUDA graph 配置在两个 cell 间漂移。

### 9.2 执行前：先判断“这个动作能不能产生外部副作用”

执行前边界位于模型或 Harness 已经提出动作之后、工具或环境尚未改变之前：

```text
候选 tool/environment action
    -> parser 与类型校验
    -> action mask、权限、预算、路径与风险检查
    -> 通过后才执行
```

当前真实 GSM8K bridge 采用 `no-tools-single-turn-v1`，`remaining_tool_calls=0`，所以它在这个边界上选择完全不执行工具。calculator smoke 另有受限 AST evaluator，但不能把 calculator 的安全性质自动推广到 shell、浏览器或 Hermes 工具。

Hermes 会在自己的 runtime 内执行工具。若未来用 RL 更新这些控制动作，必须在 Hermes 真正调用工具之前暴露并检查 action，不能只在工具已经执行后读取 metadata。

### 9.3 写入前：先判断“这条数据能不能成为训练事实”

artifact 一旦写入并进入训练队列，就可能影响梯度。因此写入前要检查：

```text
request 与 episode identity 唯一
prompt、token、log-prob、mask、reward 与 version 对齐
Harness checkpoint 可以重放 decision
runtime contract、bridge 和 score hash 一致
路径位于允许根目录
文件私有、无 symlink、无敏感 key
```

当前项目的 bridge writer 在打开目标文件前再次运行完整 validator，使用独占创建与私有权限。C0/C1 cell 在停止 GPU monitor 和 redactor 后做最终 tree audit，并用 tree digest 检测审计后的任何内容或权限变化。

这个边界不能放到 optimizer 之后。坏 log-prob、错误 version 或基础设施失败若已经进入 advantage 和梯度聚合，事后删除 JSON 也无法撤销 optimizer state 的改变。

### 9.4 发布前：先判断“候选能不能替换活动系统”

两个 optimizer 都成功，只能说明产生了两个 candidate。发布前还要检查：

```text
两个 candidate 都来自同一个 parent release
candidate version 与 JointVersion 一致
checkpoint、optimizer state 与内容 hash 完整
独立验证集、历史回归和安全测试通过
expected_active_release_id 仍是准备 candidate 时的父版本
```

前四项中，当前 `JointReleaseStore` 已实现版本、hash、父 release CAS 和原子活动指针；独立任务回归门仍属于目标设计，尚未接到真实 optimizer candidate。

发布前边界应由被优化系统不可写的评价证据控制。否则 policy 与 Harness 可以共同学会迎合自己的训练 evaluator，再用同一 evaluator 给自己放行。

## 10. 为什么 C0/C1 与 C2 都只是数据面诊断

C0/C1 的目标非常窄：判断 SGLang 生成端报告 log-prob 的公式是否导致 stored log-prob 与同 controller 重算值不一致。

两个 cell 固定：

```text
模型 checkpoint
Harness controller
GSM8K 未见样本位置 [32, 36)
seed 与采样配置
GPU、SGLang server args 和依赖版本
原有 mean <= 0.02 / max <= 0.10 概率门
```

唯一处理差异是：

```text
C0: standard log(softmax(logits))
C1: SGLANG_RETURN_ORIGINAL_LOGPROB=1, 使用 original log_softmax(logits)
```

比较器还要求：

1. 两边 effective prompt、完整 Harness state/checkpoint/decision 和输出 token 相同。
2. 两边 rescored log-prob 最大差不超过 `1e-6`。
3. 比较 ratio error 时统一使用 C0 rescored log-prob 作为共同 target。
4. C1 stored log-prob 确实发生变化。
5. C1 每条轨迹不劣于 C0，至少一条发生真实改善，并且 4/4 通过原门。

C0/C1 的实际结果是 `2/4` 通过，C1 stored log-prob 的变化量只有 `1.19e-7` 到 `4.77e-7`，两条失败轨迹的 max error 仍为 `0.12716` 和 `0.11109`。因此 `mechanism_supported=false`。这个实验没有完成以下任何一步：

```text
没有计算 policy advantage
没有计算 Harness advantage
没有调用 actor.ppo_update()
没有调用 Harness optimizer
没有生成真实 policy candidate
没有生成真实 Harness candidate
没有切换联合 active release
```

它回答的是“rollout 概率数据能不能被信任”，不是“模型和 Harness 能不能学会”。

把校准误写成联合更新，会混淆三种完全不同的事件：

```text
compute_logp()       重新计算概率，不改参数
optimizer update     根据 loss 改 candidate 参数
joint publish        让完整 candidate 二元组变成活动版本
```

当前 C0/C1 只执行第一类事件。

C2 同样没有进入后两类事件。它还暴露了一个更早的问题：两边 output token 全部不同，导致 `score_alignment=false`，无法形成逐 token common-target paired metrics。配置门通过只能证明 treatment 控制正确，不能替代结果的可比性检查。

## 11. 从当前代码走到第一次真实联合更新，还差哪些最小步骤

下面是按依赖顺序排列的最小闭环，不是已经完成的功能清单。

### 步骤 1：先让数据面的比较对象可识别

C0/C1 已经以 `2/4` 和 `mechanism_supported=false` 结束。C2 也已完成，但四条 output token 全部不同，common-target paired metrics 为 `null`，同样是 `mechanism_supported=false`。两轮都没有解锁 optimizer，也没有解锁预注册的 32 条校准与 32 条封存确认。

进入下一轮运行前，应先选择可识别的 estimand，并据此设计确定性 replay 或分布级实验。不能先决定运行 C3，再事后为已有输出寻找问题定义。

### 步骤 2：构造真实 `FrozenJointBatch`（Q/R/S 已完成数据对象）

`areal_policy_admission.py`、`harness_action_admission.py` 与 `joint_credit_alignment.py` 已把通过审计的 AReaL 原生 tensor sample 和 Harness action 通过 P record/model-call ID 接合，写入完整 Q/R admission、批次 digest、两路 estimator provenance，并执行 lag0 admission。`individual` 与 `concat` 都会从 JSON 持久 record 重新验证；post-batch 绑定、跨 episode/P record、混合版本、缺失 target 与 mask 错位会失败。

必须加入负向测试：

- score 与 bridge request 错配。
- policy token version 混合。
- Harness controller version 错配。
- policy 与 Harness episode 集不同。
- credit target 缺失、重复或跨类型。
- audit 后 artifact 被修改。

当前步骤 2 的边界是“冻结 optimizer-ready 数据”，不是“optimizer 已运行”。`policy_optimizer_update` 与 `harness_optimizer_update` 仍固定为 `false`。

### 步骤 3：接入真实 policy optimizer，但暂不发布

复用 AReaL `actor.compute_advantages()` 与 `actor.ppo_update()`，产出一个可恢复的 policy candidate checkpoint。先验证：

```text
mask=1 的 policy credit 能改变 policy candidate
只改 Harness credit 时 policy candidate 不变
mask=0 token 的任何变化都不能改变 policy candidate
```

### 步骤 4：实现生产 Harness optimizer，但暂不发布

把 tabular controller 的语义迁移到 batched Torch controller，保留 state、action mask、old log-prob、behavior version 和独立 optimizer state。

先验证：

```text
mask=1 的 Harness credit 能改变 Harness candidate
只改 policy credit 时 Harness candidate 不变
mask=0 decision 的任何变化都不能改变 Harness candidate
```

### 步骤 5：生成完整联合 checkpoint

联合 checkpoint 至少要保存：

```text
parent release 与 candidate JointVersion
policy model + optimizer + scheduler state
Harness model + optimizer state
各自 RNG state
联合 rollout cursor 与 batch digest
数据加载位置和必要的分布式恢复元数据
```

当前 toy `JointCheckpoint` 只保存两个小参数向量、toy momentum、RNG 和 cursor。真实接入不能把它原样当作 AReaL checkpoint。

### 步骤 6：通过独立门后原子发布

只有两个 candidate 都完成 checkpoint、干预测试、held-out 评价、历史回归和安全检查，才构造新的联合 manifest。活动指针切换后，再启动使用新 `(P_(k+1), H_(k+1))` 的 rollout。

第一次真实成功记录至少应当包含：

```text
父 release ID 与新 release ID
冻结 batch digest
policy/Harness optimizer step: k -> k+1
两个 candidate 的内容 hash
发布门报告
发布故障恢复结果
新版本 rollout 的 JointVersion
```

缺少其中任一关键证据时，应降低结论强度，而不是用“训练跑完”概括。

## 12. 检查题

先自己回答，再看后面的参考答案。

### 题 1

policy 与 Harness 的 optimizer 先后串行执行，能否仍称为“同时更新”？需要满足什么条件？

### 题 2

某个 token 的 `policy_release_id=P7`，但 `inference_engine_version=8`，而冻结父版本声明 engine version 7。它能进入 batch 吗？

### 题 3

Harness 动作 `REPLAN` 的原始 logit 最大，但 `action_mask=False`。计算 `old_harness_logprob` 时应不应该把它放进 softmax 分母？

### 题 4

一次 episode reward 是 1.0。能否直接令所有 policy token advantage 和所有 Harness decision advantage 都等于 1.0？

### 题 5

AReaL `actor.ppo_update()` 成功并把新模型权重同步给 rollout，但 Harness optimizer 失败。活动版本应是什么？

### 题 6

磁盘上已经出现 policy candidate 与 Harness candidate 两个对象，是否等于联合发布成功？

### 题 7

为什么 Hermes 已经有工具调用和循环控制，仍不能直接说 Harness 被 RL 更新了？

### 题 8

C2 的配置门通过，而且 C2b 在部分轨迹上的 observed max error 比 C2a 更低。为什么仍不能说关闭普通 CUDA Graph 修复了概率一致性？

### 题 9

为什么写 artifact 后、optimizer 前再做校验，仍然可能太晚？

### 题 10

一个跨越发布点的 episode 内部始终使用旧 `(P7,H3)`，记录也自洽。lag0 consumer 在活动版本已经是 `(P8,H4)` 时应怎样处理它？

## 13. 参考答案

1. 可以。两边必须读取同一冻结父版本和同一 episode 集，只产生 candidate，并且最终只成对发布。这里的“同时”是联合事务语义，不是墙钟上的同一微秒。
2. 不能。发布身份与引擎实际版本不一致，说明采样来源没有被唯一确定。应整条拒绝，而不是把 version 改成 7。
3. 不应该。`action_mask=False` 的动作概率必须为 0，不能进入 masked softmax 分母。否则记录的 old log-prob 不对应真实可选动作空间。
4. 不能直接这样做。reward 是终局结果，advantage 是动作级 credit。两类动作至少需要明确的归因规则、baseline、mask 和 estimator version。教学上的复制不能冒充已实现的估计器。
5. 继续使用旧完整版本 `(P7,H3)`。新 policy 权重只能作为未发布 candidate 保存。如果已经单独同步到活动 rollout，说明发布边界设计错误，应停止采样并恢复旧二元组。
6. 不等。对象只是 candidate 内容。只有 manifest 校验通过且 `active` commit point 成功切换，才算发布。
7. 因为“执行了 Harness 逻辑”不等于“把 Harness 动作表示成有行为概率、版本和 credit 的训练样本”。当前 Hermes 示例主要把 LLM 调用接给 AReaL 的模型 PPO，没有第二个 Harness optimizer。
8. 因为四条 output token 全部不同，`generation_equal=false`、`score_alignment=false`。两边的 observed error 属于不同上下文和不同 token，common-target paired metrics 只能是 `null`。配置门通过证明的是单变量控制成立，不证明 treatment effect 可识别，因此不能声称修复或恶化，也不能解锁 32 条校准、32 条封存确认或 optimizer。
9. 如果写入动作本身会覆盖既有事实、泄漏凭据或越过允许路径，事后校验无法撤销副作用。正式 writer 应先验证内容和路径，再以独占方式写入。
10. 拒绝为 stale。它内部自洽，只说明它是合法旧版本轨迹；lag0 还要求行为 `JointVersion` 等于当前活动版本。

## 14. 本课小结

从可信 rollout 到真实联合更新，中间不是一个 `optimizer.step()`，而是六个连续对象：

```text
可信 interaction
    -> 冻结联合 batch
    -> 两路动作级 credit
    -> policy candidate
    -> Harness candidate
    -> 联合验证
    -> 原子活动版本切换
```

其中任何一步缺失，结论都要相应降低：

- 只有 rollout 与 score：数据面检查。
- 有真实 policy optimizer，没有 Harness optimizer：Harness-native policy RL。
- 有 toy 双更新：控制面语义验证。
- 有两个真实 candidate，但没有联合发布：候选联合训练，不是活动版本更新。
- 两个真实 optimizer、联合 checkpoint、独立发布门和原子切换全部完成：才是本项目所说的“真正同时更新 policy 与 Harness”。

下一步最重要的不是立即命名 C3，而是先把可识别的 estimand 写清楚：究竟要用确定性 replay 比较同一 token 的概率偏差，还是要研究 CUDA Graph 对生成分布的影响。没有对齐的比较对象，后面的 PPO ratio 即使能算，也没有可靠含义。
