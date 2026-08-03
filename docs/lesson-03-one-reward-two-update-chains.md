# 第三课: 一份 episode reward, 为什么必须形成两条更新链

本课只回答一个核心问题:

> 一个 episode 最后只有一个总奖励, 为什么不能把它直接塞给一个 loss, 而必须分别更新 policy token 与 Harness decision?

先给结论。总奖励可以只有一个, 但被优化的动作有两类:

1. policy 选择下一个 token。
2. Harness 选择下一步控制动作, 例如直接回答、调用工具、验证、重规划或压缩上下文。

两类动作的样本位置、行为概率、可训练掩码、信用归属和参数集合都不同。因此, 同一个 episode reward 必须先被转换成两组与动作一一对应的 credit, 再进入两条彼此独立、最终同步发布的更新链。

本文的 synthetic 控制面例子对应项目提交 `01909d15b672b040ea16b9afc8b875d067e63b18`；文末证据边界仍按该实验解释。当前项目已经额外完成 Q/R/S 的真实样本准入与冻结 credit 数据对象，但仍没有真实 optimizer update。这个区分很重要，因为“credit 已对齐”和“参数已更新”是两个不同证据层级。

## 1. 先划清三个证据层级

| 层级 | 输入来自哪里 | 更新了什么 | 当前状态 |
| --- | --- | --- | --- |
| synthetic CPU fixture | 本地确定性构造的 trace、credit 和版本事件 | toy policy 参数与 toy Harness 参数 | 已实现并有测试证据 |
| 真实 AReaL Q/R/S admission | AReaL 推理引擎产生的 token、logprob、输出版本，加同一 episode 的真实 Harness 决策与两份冻结 baseline | 两路 optimizer-ready 样本与 advantage，不执行更新 | 已实现，并通过 `individual/concat`、lag0、mask 与持久 record 重验 |
| 真实 optimizer update | 真实 policy 模型参数、真实 Harness 可学习参数及各自 optimizer state | policy 与 Harness 的生产级联合学习 | 尚未完成 |

这里的 `sidecar` 是和主训练样本并排保存的一组附加字段。早期 `JointDecisionBatch` 只记录 policy token 与 Harness decision 的决策和 synthetic credit，不是完整 AReaL PPO batch。现在 `areal_policy_admission.py` 保留六字段 AReaL sample 和精确 decision span，`harness_action_admission.py` 保留行为时 Harness state/distribution，`joint_credit_alignment.py` 把两者与同一 P record 和 `JointVersion` 持久连接。它们已经形成完整训练输入，但不包含任何 optimizer 执行结果。

因此, 本课讲的不是一次已经完成的真实 AReaL 联合训练。它讲的是: 当前提交已经把两条更新链的边界、版本约束、失败条件和本地最小闭环写成了可执行契约。

## 2. 从可计算对象开始

### 2.1 一个 episode reward

设一次完整交互为 episode (e), 结束后环境或评价器给出一个标量奖励:

\[
R_e \in \mathbb{R}.
\]

`R_e` 只评价整个 episode 的结果。它没有直接说明哪个 token 应当负责, 也没有直接说明哪个 Harness 决策应当负责。

### 2.2 policy 动作

在第 (t) 个生成位置, policy 的输入是已有上下文 (x_t), 输出是下一个 token 的概率分布:

\[
\pi_\theta(\cdot \mid x_t): \mathcal{V} \to [0,1],
\qquad
\sum_{v\in\mathcal{V}} \pi_\theta(v \mid x_t)=1.
\]

其中:

- \(\mathcal{V}\) 是 tokenizer 的有限词表。
- \(a_t \in \mathcal{V}\) 是实际采样的 token。
- \(\theta\) 是 policy 参数。

policy 的一个训练目标对应一个确切 token 位置, 不是整个文本的模糊标签。

### 2.3 Harness 动作

在第 (d) 个控制位置, Harness 观察运行状态 (z_d), 从当前允许的动作中选择一个:

\[
h_\phi(\cdot \mid z_d, m_d): \mathcal{A}_H \to [0,1].
\]

其中:

- \(\mathcal{A}_H\) 是 Harness 动作集合。
- (m_d(a)\in\{0,1\}) 表示动作 (a) 在这个状态是否允许。
- 对所有 (m_d(a)=0) 的动作, 采样概率必须为 0。
- \(u_d\in\mathcal{A}_H\) 是实际选中的 Harness 动作。
- \(\phi\) 是 Harness 参数。

当前项目中的动作类型包括 `DIRECT`、`RETRIEVE_SKILL`、`VERIFY`、`REPLAN` 和 `COMPRESS`。动作集合的顺序由具体样本里的 `action_ids` 决定, 不能依赖枚举名称猜测索引。

### 2.4 一份奖励不等于一个训练样本

假设一个 episode 含有 120 个可训练 token 和 4 个可训练 Harness 决策。虽然只有一个 (R_e), 优化器看到的其实是 124 个有位置的动作。训练前至少要回答:

1. 哪些动作参与 loss?
2. 每个动作分到多少 credit?
3. 动作由哪个旧版本采样?
4. 动作在旧版本下的概率是多少?
5. 更新后 policy 与 Harness 如何作为一个完整版本对外可见?

这五个问题分别对应样本字段、两类 loss mask、credit、行为版本和联合发布。

## 3. 当前代码中的两种样本

实现位于 `jphrl/trajectory/joint_batch.py`。

### 3.1 `PolicyTokenSample`

每个对象只描述一个输出 token:

| 字段 | 直接含义 |
| --- | --- |
| `episode_id` | token 属于哪个 episode |
| `model_call_id` | token 属于哪次模型调用 |
| `output_position` | token 在该次模型输出中的位置 |
| `token_id` | 被采样 token 的词表整数 ID |
| `old_policy_logprob` | 行为 policy 采到该 token 时的对数概率 |
| `policy_loss_mask` | 该 token 是否进入 policy loss, 只能是 0 或 1 |
| `policy_release_id` | 产生该 token 的 policy 发布 ID |
| `inference_engine_version` | 推理引擎为该 token 记录的权重版本 |
| `advantage` | 分配给该 token 所属模型调用的 credit 数值 |
| `credit_source` | 该 credit 由哪个过程计算 |

唯一位置键是 `(episode_id, model_call_id, output_position)`。同一个键出现两次会被拒绝。

### 3.2 `HarnessActionSample`

每个对象只描述一个 Harness 决策:

| 字段 | 直接含义 |
| --- | --- |
| `episode_id` | 决策属于哪个 episode |
| `decision_id` | 决策的唯一 ID |
| `action` | 实际选中的动作 |
| `action_ids` | 当时参与离散选择的动作列表 |
| `action_mask` | 当时每个动作是否允许 |
| `pre_mask_logits` | 屏蔽非法动作前的未归一化分数 |
| `old_harness_logprob` | 应用 `action_mask` 后, 选中动作的旧对数概率 |
| `harness_loss_mask` | 该决策是否进入 Harness loss, 只能是 0 或 1 |
| `harness_behavior_version` | 产生该决策的 Harness controller 版本 |
| `advantage` | 分配给该决策的 credit 数值 |
| `credit_source` | 该 credit 由哪个过程计算 |

唯一位置键是 `(episode_id, decision_id)`。代码还要求 policy 的 `model_call_id` 与 Harness 的 `decision_id` 不得交叉复用, 防止 credit 被投到错误动作类型。

### 3.3 `JointDecisionBatch`

`JointDecisionBatch` 把两条流放在同一个版本边界内:

```text
JointDecisionBatch
├── joint_version
├── episode_ids
├── policy_tokens: tuple[PolicyTokenSample, ...]
└── harness_actions: tuple[HarnessActionSample, ...]
```

它的关键含义不是把两类动作混成一个 loss, 而是让它们共享同一个 `JointVersion`, 同时保持样本集合分离。

生产模式下, builder 会拒绝未完成或无效的 trace。也就是 reward 缺失、success 缺失、`trace.valid` 为假或没有 `episode_ended` 事件时都会失败。G1 为了构造局部 synthetic fixture, 明确传入 `allow_open_fixtures=True`。这个开关不能被误读成生产训练允许不完整 episode。

## 4. 两个 loss mask, 以及一个不能混淆的 action mask

这部分最容易出错。代码里有三个名字带 `mask` 的对象, 但只有前两个决定样本是否进入 loss。

### 4.1 policy loss mask

对 token (t):

\[
M_t^\pi \in \{0,1\}.
\]

- (M_t^\pi=1): token 可以贡献 policy loss。
- (M_t^\pi=0): token 保留在轨迹中用于对齐或审计, 但不能改变 policy 参数。

### 4.2 Harness loss mask

对 Harness 决策 (d):

\[
M_d^H \in \{0,1\}.
\]

- (M_d^H=1): 决策可以贡献 Harness loss。
- (M_d^H=0): 决策保留记录, 但不能改变 Harness 参数。

### 4.3 Harness action mask

对候选动作 (a):

\[
m_d(a) \in \{0,1\}.
\]

它回答的是"这个动作当时能不能选", 不是"这条已经发生的决策要不要训练"。当前校验器要求:

- `action_mask` 每个值都必须是布尔值。
- 至少一个动作允许。
- 被选中的 `action` 必须存在于 `action_ids` 且对应 mask 为真。
- `action_ids`、`action_mask` 和 `pre_mask_logits` 长度必须相等。

一句话区分:

> `action_mask` 约束采样空间, `policy_loss_mask` 和 `harness_loss_mask` 约束梯度空间。

## 5. credit 的 target 与 source

### 5.1 target 是数值贴到哪个动作上

当前代码用两张不同的映射表表示 target:

```python
EpisodeCredit(
    policy_calls: Mapping[model_call_id, DecisionCredit],
    harness_decisions: Mapping[decision_id, DecisionCredit],
)
```

`policy_calls` 中的数值只能投给同名模型调用的 token。`harness_decisions` 中的数值只能投给同名 Harness 决策。两张表分开, 是为了让类型错误尽早失败。

### 5.2 source 是数值由哪个计算过程产生

每个 `DecisionCredit` 包含:

```python
DecisionCredit(advantage: float, source: str)
```

- `advantage` 是有限实数, 表示该动作相对某个参照值好多少。
- `source` 是非空字符串, 记录这个数由哪个过程计算。

当前 G1 使用的 source 名称是:

- policy: `synthetic-policy-credit-fixture-v1`
- Harness: `synthetic-harness-credit-fixture-v1`

这些名字明确表示它们是 synthetic fixture。它们不是已经实现的 verifier credit, 也不是已经实现的 counterfactual credit。

### 5.3 source 不同仍然不足以证明更新链分离

把两个 source 字符串写成不同名字, 只能证明元数据不同。把两个 advantage 写成不同数值, 也只能证明输入值不同。真正更强的检查是干预:

1. 只改变 policy credit, Harness checkpoint 必须不变。
2. 只改变 Harness credit, policy checkpoint 必须不变。
3. 只扰动 mask 为 0 的样本, 两边 checkpoint 都必须不变。

这正是当前 G1 的 8 个干预不变性检查在做的事情。

## 6. 逐步数字例子: 同一个奖励怎样进入两条链

下面的数字用于教学, 不是声称 G1 已实现真实 advantage estimator。

### 第一步: episode 得到一个总奖励

假设:

\[
R_e=1.00.
\]

这个 episode 中有一次模型调用和一次 Harness 决策。

### 第二步: 分别计算两类 credit

为了演示, 给两类动作使用不同参照值:

\[
A^\pi = R_e-b^\pi = 1.00-0.25=0.75,
\]

\[
A^H = R_e-b^H = 1.00-0.60=0.40.
\]

`A^pi` 的 target 是某个 `model_call_id`, `A^H` 的 target 是某个 `decision_id`。它们共享奖励来源, 但不是同一个训练样本。

### 第三步: 构造 policy token 流

假设模型输出两个 token:

| token | `token_id` | `output_position` | `old_policy_logprob` | `policy_loss_mask` | `advantage` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 205 | 0 | -0.20 | 1 | 0.75 |
| 1 | 99 | 1 | -0.80 | 0 | 0.75 |

当前 toy updater 为每个 policy token 计算两个特征:

\[
f^\pi(t)=
\left(
\frac{(\text{token\_id}\bmod17)-8}{8},
\frac{(\text{output\_position}\bmod3)-1}{2}
\right).
\]

对 token 0:

\[
205\bmod17=1,
\qquad
f^\pi(0)=(-0.875,-0.5).
\]

所以它对 toy policy 梯度和的贡献是:

\[
M_0^\pi A^\pi f^\pi(0)
=1\times0.75\times(-0.875,-0.5)
=(-0.65625,-0.375).
\]

token 1 的 `policy_loss_mask=0`, 即使把它的 `token_id` 或 `advantage` 改得很大, 贡献仍然严格为 0。

### 第四步: 构造 Harness decision 流

假设样本自己的 `action_ids` 有 5 个动作, 选中索引 3, 且 `harness_loss_mask=1`。当前 toy updater 使用:

\[
f^H(d)=
\left(
\frac{i-(K-1)/2}{K},
\begin{cases}
1,&i\text{ 为偶数}\\
-1,&i\text{ 为奇数}
\end{cases}
\right),
\]

其中 (K=5), (i=3)。因此:

\[
f^H(d)=(0.2,-1).
\]

它对 toy Harness 梯度和的贡献是:

\[
M_d^H A^H f^H(d)
=1\times0.40\times(0.2,-1)
=(0.08,-0.40).
\]

### 第五步: 看清没有交叉项

toy updater 实际形成的是两个独立求和:

\[
g_\pi=\frac{1}{N_\pi}\sum_t M_t^\pi A_t^\pi f^\pi(t),
\]

\[
g_H=\frac{1}{N_H}\sum_d M_d^H A_d^H f^H(d).
\]

其中 (N_\pi) 和 (N_H) 分别是 mask 为 1 的 policy token 数和 Harness 决策数。不存在 `Harness advantage × policy token feature`, 也不存在 `policy advantage × Harness action feature`。

若某一条流的有效样本数为 0, 当前 toy updater 会原样返回该组件, 不增加它的 optimizer step。这里的更新仅用于检查分流和状态转换, 不是 PPO, 也不是生产级 Harness optimizer。

## 7. old logprob 与 behavior version 分别解决什么问题

### 7.1 old logprob 记录动作当时有多可能

定义:

\[
\ell_{\text{old}}=\log p_{\text{behavior}}(a\mid s).
\]

它是采样动作时的行为分布概率的自然对数。policy token 使用 `old_policy_logprob`, Harness 决策使用 `old_harness_logprob`。

在真实 PPO 类更新中, 新旧概率比通常写成:

\[
r=\exp(\ell_{\text{new}}-\ell_{\text{old}}).
\]

例如旧 logprob 为 -0.20, 新 logprob 为 -0.10, 则:

\[
r=\exp(0.10)\approx1.105.
\]

这表示新参数下该动作的概率约为行为参数下的 1.105 倍。policy 与 Harness 必须各自保存 old logprob, 因为它们来自两个不同分布。

当前 toy updater 不用 old logprob 计算 PPO ratio。它只通过 batch 校验把 old logprob 纳入来源审计。因此, 看到字段存在不能推导出真实 PPO 已经执行。

### 7.2 policy release ID 与 inference engine version 不是一回事

对每个 policy token, 当前契约同时记录:

- `policy_release_id`: 这次模型响应声明属于哪个已发布 policy。
- `inference_engine_version`: 推理引擎实际为这个 token 标记的权重版本整数。

前者必须等于 batch 的 `JointVersion.policy`。后者必须是非负整数, 且模型输出中的版本列表长度必须与输出 token 数相同。

当前 runner 会从 `ModelResponse` 写入这两个字段。本地静态 `HuggingFaceChatModel` fixture 使用全 0 的 token 版本列表, 这只表示本地静态 fixture, 不表示真实 AReaL 在线版本已经接通。

对 Harness, `harness_behavior_version` 来自决策事件的 controller 版本, 并且必须等于 `JointVersion.harness_controller`。

## 8. `JointVersion` 与 lag0 admission

`JointVersion` 是一次交互所依赖的联合版本记录。除 policy 和 Harness controller 外, 它还包含 Harness artifact、tool schema、parser、environment、evaluator、tokenizer 和 context builder 等版本。

当前严格准入函数可以直接写成:

```python
trace.validate()
if trace.joint_version != active_joint_version:
    raise StaleJointVersionError(...)
```

这叫 lag0 admission。这里的 lag 是二元组:

\[
(L_\pi,L_H)=(0,0).
\]

- (L_\pi=0): 不接受旧 policy 产生的 trace。
- (L_H=0): 不接受旧 Harness controller 产生的 trace。

它不是只比较 policy 版本或只比较 Harness 版本, 而是要求整个 `JointVersion` 完全等于当前活动版本。这样可以拒绝 policy 新而 Harness 旧、policy 旧而 Harness 新, 以及工具或解析器版本错配的轨迹。

当前 G1 的版本调度是 synthetic CPU fixture, 不是生产 AReaL consumer。该 fixture 构造 1000 条确定性版本样本, 做 10 次联合发布, 其中 100 条跨越发布边界。lag0 拒绝这 100 条过期样本, 接受的过期样本数为 0。

## 9. 8 个干预不变性检查

干预的定义是: 只改变一个指定输入, 然后重新计算输出, 观察不应受影响的组件是否保持完全相同。

当前 G1 检查以下 8 条:

| 编号 | 只改变什么 | 必须观察到什么 |
| ---: | --- | --- |
| 1 | mask 为 1 的 policy credit | policy checkpoint 改变 |
| 2 | 同上 | Harness checkpoint 不变 |
| 3 | mask 为 1 的 Harness credit | Harness checkpoint 改变 |
| 4 | 同上 | policy checkpoint 不变 |
| 5 | mask 为 0 的 policy token ID 与 credit | policy checkpoint 不变 |
| 6 | 同上 | Harness checkpoint 不变 |
| 7 | mask 为 0 的 Harness credit | Harness checkpoint 不变 |
| 8 | 同上 | policy checkpoint 不变 |

这 8 条一起回答两个问题:

1. 每条有效 credit 确实能到达自己的组件。
2. credit 不会串到另一个组件, 被 mask 的样本也不会漏进任何更新。

它们比检查 source 字符串不同更强, 因为它们直接比较更新结果。

## 10. 10 个负向 mutation

负向 mutation 是故意把一份合法样本改坏一处, 然后要求校验器拒绝它。当前 G1 覆盖 10 种错误:

1. policy logprob 列表长度与输出 token 数不一致。
2. policy loss mask 出现 0、1 之外的值。
3. `policy_release_id` 与 batch 版本不一致。
4. inference engine version 列表长度与输出 token 数不一致。
5. Harness 选中了 `action_mask=False` 的动作。
6. `old_harness_logprob` 与 masked logits 重新计算的结果不一致。
7. `harness_behavior_version` 与 batch 版本不一致。
8. policy call ID 与 Harness decision ID 交叉投放 credit。
9. 同一 trace 内的事件携带混合版本。
10. 生产 builder 收到没有正常结束的 open trace。

这 10 条证明的是输入契约会拒绝已枚举的坏数据。它们不证明所有未知错误都已经被覆盖。

## 11. 为什么两条更新链还要 atomic pair publish

假设 policy 已经从 (P_0) 更新到 (P_1), Harness 已经从 (H_0) 更新到 (H_1)。如果两个组件分别切换, 读者可能看到不存在于训练计划中的半组合:

\[
(P_1,H_0) \quad\text{或}\quad (P_0,H_1).
\]

atomic pair publish 的要求是读者只能看到:

\[
(P_0,H_0) \quad\text{或}\quad (P_1,H_1).
\]

当前 `JointReleaseStore` 的本地实现包含:

1. policy 与 Harness 的内容寻址对象。
2. 引用完整二元组的 release manifest。
3. `active.json` 活动指针。
4. 本地 `fcntl` 文件锁。
5. 基于 expected active release ID 的 compare-and-swap 检查。
6. 用临时文件、`fsync` 和 `os.replace` 完成活动指针切换。

故障矩阵覆盖:

- policy 对象写完后失败。
- Harness 对象写完后失败。
- 活动指针切换前失败。
- 活动指针切换后失败。
- 无故障。
- release gate 拒绝。

切换前失败时可能留下未被引用的对象或 manifest, 但活动读者仍看到旧的完整二元组。切换成功后, 读者看到新的完整二元组。

证据边界必须说清: 当前证明的是本地 POSIX 文件系统上的 commit point 故障矩阵。它不是分布式对象存储一致性证明, 也不是机器突然断电下所有硬件和文件系统组合的普遍证明。

## 12. toy checkpoint 保存了什么

### 12.1 组件 checkpoint

`ComponentCheckpoint` 分别用于 policy 与 Harness, 保存:

- `version`
- `parameters`
- `optimizer_momentum`
- `optimizer_step`
- `rng_state`
- `sample_count`
- `state_sha256`

`state_sha256` 是组件状态内容的哈希, 读取时会重新计算并比对。

### 12.2 联合 checkpoint

`JointCheckpoint` 的 schema 是 `jph.joint-checkpoint.v1`, 保存:

- 完整 `JointVersion`
- `active_release_id`
- `macro_step`
- policy 组件 checkpoint
- Harness 组件 checkpoint
- 联合 `rng_state`
- `rollout_cursor`

写入磁盘时, 外层 envelope 的 schema 是 `jph.joint-checkpoint-envelope.v1`, 并包含 `checkpoint_sha256`。

`validate_checkpoint` 还会把 checkpoint 与 release store 当前活动发布交叉检查, 包括 active release ID、联合版本、policy payload 与 Harness payload。

当前 G1 完成的是 toy one-step replay: 写入初始 checkpoint, 读取恢复, 分别从连续状态与恢复状态计算下一步, 再比较结果。它没有保存真实 AReaL 模型和 optimizer 的完整状态, 也没有覆盖 CUDA RNG、分布式通信、队列、数据加载游标等生产状态。

因此, `toy checkpoint` 不能改名为 `AReaL training checkpoint`。

## 13. 当前 G1 到底证明了什么

当前确定性 CPU fixture 构造:

- 32 条 synthetic decision trace。
- 64 个 policy token, 其中 48 个可训练。
- 32 个 Harness action, 其中 24 个可训练。
- 1000 条确定性版本 fixture。
- 10 次联合发布与 100 条跨发布边界样本。
- 8 个更新干预不变性检查。
- 10 个负向 mutation。
- 本地发布故障矩阵。
- toy one-step checkpoint replay。

它能支持的结论是:

> 在当前提交的确定性 synthetic CPU 控制面 fixture 中, policy token 与 Harness decision 有分离的数据、mask、credit 和 toy 更新路径; 版本错误与列出的坏输入会被拒绝; policy 与 Harness 的本地发布以完整二元组切换; toy checkpoint 可做一步确定性恢复比较。

它不能支持的结论是:

- 已经完成 AReaL policy update。
- 已经完成生产级 Harness update。
- 已经观察到真实 policy 与 Harness 联合学习收益。
- 已经证明真实分布式训练在故障下可恢复。

代码中的 claim boundary 也明确写着: 该运行只覆盖确定性 synthetic CPU 控制面 fixture、本地 POSIX commit-point 故障矩阵和 toy one-step replay; 它不执行 AReaL policy update, 也不声称已经实现 policy/Harness joint learning。

## 14. 把整条链压缩成一张图

```text
episode trace + R_e
        |
        +--> policy credit map, target=model_call_id
        |           |
        |           +--> PolicyTokenSample
        |                    |
        |                    +--> policy_loss_mask
        |                    +--> old_policy_logprob
        |                    +--> policy_release_id
        |                    +--> inference_engine_version
        |                              |
        |                              +--> policy updater --> P_1
        |
        +--> Harness credit map, target=decision_id
                    |
                    +--> HarnessActionSample
                             |
                             +--> harness_loss_mask
                             +--> action_mask + logits
                             +--> old_harness_logprob
                             +--> harness_behavior_version
                                       |
                                       +--> Harness updater --> H_1

JointVersion lag0 admission: 两条输入都必须属于活动版本
atomic pair publish: 只发布完整的 (P_1, H_1)
joint checkpoint: 保存并校验这个完整二元组的 toy 状态
```

## 15. 检查题

先自己回答, 再看题后参考答案。

### 题 1

一个 episode 只有一个 `R_e`, 是否意味着 policy 与 Harness 应使用同一个 `advantage` 数值? 为什么?

### 题 2

某个 Harness 动作在 `action_mask` 中为 `False`, 但 `harness_loss_mask=1`。这个样本是否合法?

### 题 3

把一个 `policy_loss_mask=0` 的 token 的 `advantage` 从 0.1 改成 100.0, policy checkpoint 改变了。这说明哪里有 bug?

### 题 4

为什么仅仅看到 policy 与 Harness 的 `credit_source` 字符串不同, 还不能证明两条更新链已经分离?

### 题 5

trace 的 policy 版本是当前版本, Harness controller 是上一个版本。lag0 admission 是否应该接受?

### 题 6

为什么 `policy_release_id` 与 `inference_engine_version` 都需要记录? 两者分别回答什么问题?

### 题 7

atomic pair publish 在活动指针切换前失败, 磁盘上可能已有新 policy 对象。此时读者应该看到哪个组合?

### 题 8

G1 的 toy updater 产生了 policy 与 Harness 两个新版本。能否据此宣称已经完成真实 AReaL PPO 联合训练?

### 参考答案

1. 不意味着。`R_e` 是共享结果, `advantage` 是贴到具体动作 target 上的训练 credit。两类动作可以使用不同 baseline 或 credit estimator, 所以数值可以不同。
2. 不合法。`action_mask=False` 表示动作当时不可选, 已选动作必须对应允许位置。`harness_loss_mask` 不能把一次非法采样变成合法样本。
3. mask 没有在 policy 更新入口生效, 或被 mask 的样本仍参与了聚合。它违反干预检查 5。
4. 字符串只证明元数据名称不同。必须通过只改一侧 credit、观察另一侧 checkpoint 不变的干预, 才能检查实际计算依赖。
5. 不接受。lag0 要求整个 `JointVersion` 等于活动版本, 不是只要求 policy 相等。
6. `policy_release_id` 回答模型响应属于哪个发布; `inference_engine_version` 回答推理引擎为每个输出 token 标记了哪个实际权重版本。前者是发布身份, 后者是 token 级执行版本证据。
7. 应看到旧的完整 `(P_0,H_0)`。新对象可以成为暂时未引用的对象, 但不能让活动读者看到半更新组合。
8. 不能。当前 updater 是用手工特征和 toy momentum 做的确定性状态转换, 不计算真实 PPO loss, 也不更新真实模型参数。

## 16. 下一课预告: 真实 AReaL interaction bridge

下一课将从当前 sidecar 契约出发, 只处理一个新的边界: 怎样把真实 AReaL rollout 中的输出 token、old logprob、token 级 engine version 和 release ID, 与同一 episode 的 Harness decision 事件无损对齐。

我们会逐项回答:

1. AReaL 的 interaction 输出中, 哪些字段可以直接映射到 `PolicyTokenSample`?
2. 哪些字段当前缺失, 必须扩展 bridge, 而不能用默认值伪造?
3. 如何把完整 AReaL PPO tensor batch 与 `JointDecisionBatch` sidecar 通过稳定键连接?
4. rollout 跨越联合发布时, lag0 在生产 consumer 的哪个入口执行?
5. 真实 policy optimizer 与 Harness optimizer 各自需要哪些状态, 才能替换当前 toy updater?

在这些问题通过真实 trace 和集成测试之前, 项目应继续使用准确表述: sidecar 契约已存在, synthetic 控制面闭环已通过, 真实 AReaL interaction bridge 与真实 optimizer update 尚未完成。
