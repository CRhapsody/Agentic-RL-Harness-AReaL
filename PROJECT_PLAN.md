---
title: "JPH-RL：基于 AReaL 的 Policy–Harness 联合强化学习计划"
aliases: [JPH-RL, Joint Policy Harness RL, AReaL Agentic RL Harness]
type: project-plan
created: 2026-08-02
updated: 2026-08-03
status: in-progress
tags: [agentic-rl, harness, areal, joint-learning, experiment-plan]
---

# JPH-RL：基于 AReaL 的 Policy–Harness 联合强化学习计划

> [!summary] 可行性结论
> **8×A100 足以做出有研究价值的原型，AReaL 2.0 是合适底座，但不是改 YAML 即可完成。**AReaL 已经覆盖 policy rollout、token log-prob、异步训练、模型版本与权重同步；当前开源实现尚未实现 Harness/memory/skill/tool-schema 的联合演化。项目的真正工作量在第二个行为策略、联合版本、轨迹 schema、双重 staleness、复合 checkpoint、独立验收和整对回滚。

## 1. 研究目标与非目标

### 1.1 目标

构造一个 Agent，其模型 policy 与 Harness controller 从同一批在线交互证据中学习：

\[
\pi_{\theta_k}:\mathcal Z\to\Delta(\mathcal Y),
\qquad
\kappa_{\omega_k}:\mathcal S^H\to\Delta(\mathcal U).
\]

- \(\pi_{\theta_k}\) 输入实际送给 LLM 的上下文 \(z_t\)，输出文本或工具调用 token \(y_t\)。
- \(\kappa_{\omega_k}\) 输入可观测的 Harness 状态 \(s_t^H\)，输出结构化 Harness action \(u_t\)。
- \(\phi_k\) 表示跨任务持久的 prompt、skill、工具描述和 workflow artifact pool；它由慢速、受验证门约束的更新产生。

项目要验证的不是“完整系统分数变高”，而是：在相同推理和训练预算下，\(\theta\) 与 \(\omega\) 是否各自贡献收益，以及新模型与新 Harness 是否产生正交互。

### 1.2 非目标

- 不把两个 optimizer 物理上同时写参数当作贡献；联合宏步与原子发布才是可复现单位。
- 第一版不允许模型任意修改 Python Harness 代码或安全策略。
- 第一版不训练第二个大语言模型充当 Harness policy。
- 第一版不从 SWE/Terminal 长轨迹起步，避免把 sandbox 与系统故障混进算法结论。
- 不用训练 evaluator 的分数上涨替代冻结测试集上的真实任务成功。

## 2. “真正联合更新”的操作性定义

联合宏步 \(k\) 固定快照

\[
V_k=(\theta_k,\omega_k,\phi_k,
v^{tool}_k,v^{parser}_k,v^{env}_k,v^{eval}_k,
v^{tokenizer}_k,v^{context}_k).
\]

所有 episode 从开始到结束绑定同一个 \(V_k\)，并执行：

1. 从任务分布采样 \(x\)，Harness controller 选择结构动作 \(u_t\sim\kappa_{\omega_k}(\cdot\mid s_t^H)\)。
2. Harness 根据 \((u_t,\phi_k)\) 构造上下文，模型生成 \(y_t\sim\pi_{\theta_k}(\cdot\mid z_t)\)，环境返回新观察与结果。
3. 同一冻结 batch 分别计算 policy advantage 与 Harness advantage，得到 \(\theta_{k+1}\) 和 \(\omega_{k+1}\)。
4. 每隔 \(K\) 个宏步才允许提出 \(\phi\) 的受限新增、修改、删除或过期候选；候选先进入 shadow pool。
5. 两个优化器、validation、historical regression 和复合 checkpoint 全部通过后，原子发布 \(V_{k+1}\)；任一失败则整对不发布。

忽略环境中不含可训练参数的项，联合轨迹分布含

\[
p_{\theta,\omega,\phi}(\tau)
\propto
\prod_t
\kappa_\omega(u_t\mid s_t^H)
\pi_\theta(y_t\mid z_t(u_t,\phi)).
\]

相对 behavior snapshot 的单步联合比率为

\[
\rho_t=
\frac{\kappa_\omega(u_t\mid s_t^H)}
     {\kappa_{\omega_{old}}(u_t\mid s_t^H)}
\cdot
\frac{\pi_\theta(y_t\mid z_t)}
     {\pi_{\theta_{old}}(y_t\mid z_t)}.
\]

MVP 对两个分支分别 clip、分别记录 KL/entropy，并从 staleness \((0,0)\) 开始。这里只把可记录 old log-prob 的 \(\omega\) 更新称为 Harness RL；\(\phi\) 的文本/程序变异称为 validation-gated artifact evolution。

## 3. Harness MVP

### 3.1 Controller 状态

第一版只用执行时可观察且不会泄露答案的特征：turn、剩余 token/工具预算、上下文长度、最近 parser/tool 错误、检索命中、最近 verifier 状态和任务域。将类别特征 embedding 后输入 CPU 上的小型 MLP/categorical policy。

### 3.2 五个原子动作

| 动作 | 直接作用 | 成本与边界 |
|---|---|---|
| `DIRECT` | 不增加辅助信息，按标准上下文调用模型 | 最低额外成本 |
| `RETRIEVE_SKILL` | 从当前 \(\phi_k\) 检索一个有 provenance 的 skill | 受注入 token 上限约束 |
| `VERIFY` | 调用规则/工具检查当前候选或前置条件 | 计入工具与 wall-clock 预算 |
| `REPLAN` | 注入固定格式的错误状态与重规划请求 | 不直接提供正确答案 |
| `COMPRESS` | 用确定性或冻结压缩器缩短历史 | 保存压缩前 hash，便于审计 |

终止仍由模型/环境的合法动作处理，避免一个 Harness action 同时表达“压缩或终止”。retry 属于 parser/tool error 的有界状态转移，不与 `REPLAN` 混成一个动作。

### 3.3 持久 artifact

```text
HarnessArtifact {
  id, version, parent_id, content_hash,
  type: prompt | skill | tool_description | workflow_rule,
  content, provenance, compatibility,
  created_at, expires_at,
  validation_delta, regression_delta,
  status: shadow | active | rejected | retired
}
```

Artifact 不原地修改；任何变化产生新版本。首轮只允许 skill 与 tool description 的有界 patch，不改 sandbox、权限和 verifier。

## 4. AReaL 适配判断

### 4.1 已有底座

AReaL 2.0 已将训练、推理、Agent 与权重更新拆成服务；Agent workflow 可经 OpenAI-compatible proxy 发起模型调用，并捕获生成 token、log-prob、reward 和 policy version。它支持 SGLang/vLLM 推理以及 FSDP2/Megatron/Archon 训练，适合复用现有 policy 数据面。[官方仓库](https://github.com/areal-project/AReaL)、[Agentic RL 教程](https://github.com/areal-project/AReaL/blob/v2.0.0/docs/en/tutorial/agentic_rl.md)、[Online Proxy 教程](https://github.com/areal-project/AReaL/blob/v2.0.0/docs/en/tutorial/online_proxy.md)

但 v2.0.0 的原生训练 tensor contract 只有 `input_ids`、`loss_mask`、`logprobs`、`versions`、`attention_mask`、`rewards` 六项。`ModelRequest.metadata` 不会自动进入训练 tensor，`rollout.dump_to_file=true` 的 JSONL 也不含 token IDs、log-probs 或 loss mask。因此 Harness action、old Harness log-prob、controller/artifact/tool/parser/context 版本必须写入可关联的 sidecar，或显式扩展 export；不能仅凭 rollout dump 宣称联合轨迹已闭环。本项目现已实现 `model_call_id <-> interaction_id` 身份 sidecar、委托 AReaL 原生 `individual/concat` export 的可审计样本归档、多轮 Agent Service session/model-call/ready-trajectory receipt、固定 Hermes 0.19.0 的逐调用 receipt 入口、AReaL pre-batch 小型补丁，以及项目外私有 journal 的持久 exactly-once 接合。所有绑定都发生在 `export_trajectory()` 后、batch merge 前；post-batch 数据明确拒绝。Q/R 又分别完成 AReaL Policy tensor 与真实 Harness action 的 lag-zero 准入；S 将两者与同一 P record、episode 和 `JointVersion` 持久接合，并用两份独立冻结 baseline 形成逐动作 advantage 与严格 mask。Q/R/S 的 record 仍明确把两个 optimizer update 证据记为 `false`；下一阶段 T/U 才允许执行真实参数更新。

AReaL 2.0 论文进一步提出 Agent Trajectory Data Plane 与 evolution control plane，并把 Harness、memory、skill、tool 与 policy 都放进可演化对象；但论文明确把当前 prototype 的实现范围收在 policy-weight-update 分支。因此本项目是在官方愿景内补一个尚未落地的分支，不是调用现成 API。[AReaL 2.0 论文](https://arxiv.org/abs/2607.01120)

### 4.2 必须补的八个缝

| 扩展 | 最小职责 | 验收证据 |
|---|---|---|
| `HarnessRegistry` | 不可变 artifact、候选池、父子关系、hash、状态 | 任意轨迹能还原确切 artifact |
| `JointEvolutionController` | batch barrier、双 optimizer、验证、发布、回滚 | 故障注入时无半版本可见 |
| session pin | episode 开始固定联合版本 | episode 内版本切换数为 0 |
| trajectory schema | 两类 action/old log-prob、policy/controller/artifact/tool/parser/environment/evaluator/tokenizer/context 版本、reward vector | old log-prob 与 reward 可抽样复算 |
| joint batch hook | policy/Harness 分别更新并在 publish 前汇合 | 任一更新失败不前进 joint step |
| 2D staleness | \((\Delta_\pi,\Delta_H)\) 门限与丢弃/重采 | 指标按二维 lag 分桶 |
| composite checkpoint | 模型、两个 optimizer、artifact pool、RNG、游标、manifest | 恢复后下一步与对照一致 |
| trajectory journal | 任务/环境种子、工具 I/O hash、verifier 与版本 | 能复现指定失败 batch |

对 AReaL 主干应采用薄 adapter/子类和明确 schema migration，不把可变全局状态塞进并发 `agent.run()`。当前正式版本冻结为 `v2.0.0`，实际 commit `fee938eada49208a5aabdbc1095730a13076a349`；clone 后必须核验 commit，不能只记录浮动 branch。[官方 Releases](https://github.com/areal-project/AReaL/releases/tag/v2.0.0)

## 5. 环境与数据路线

### M0：确定性工具 sanity

先做纯内存 calculator，只允许整数、括号、一元正负号和 `+ - * /`，用 `fractions.Fraction` 精确计算；它不执行任意代码，因此不引入 subprocess、容器或网络。该阶段只验证版本、log-prob、reward、恢复和回滚。Python tool 另列为后续里程碑，届时才要求真实隔离 sandbox，避免把容器故障混入第一条数据面验证。

### M1：主环境 tau2

选择 tau2 airline 或 retail：它有多轮状态、结构化工具、预算权衡与可执行结果，而且 AReaL 有官方例子。[AReaL tau2 example](https://github.com/areal-project/AReaL/tree/main/examples/tau2)

固定 user simulator 版本、temperature、system prompt 和 API/本地模型；将 simulator token 与成本单独记账。早期可用小规模固定 simulator 做系统验证，但论文主实验不能把不同 simulator 混为同一环境。

### 暂缓场景

SWE/Terminal 和开放 Web 留到联合数据面稳定后。它们的环境构建、网络漂移、长轨迹、sandbox 和稀疏 reward 会使失败难以归因。

## 6. 8×A100 资源方案

### 6.1 起点

- 固定 AReaL `v2.0.0@fee938eada49208a5aabdbc1095730a13076a349`，先复现官方 8-GPU GSM8K GRPO 配置：rollout `sglang:d4p1t1`，actor `fsdp:d4p1t1`，reference 与 actor colocate。[官方 8 GPU 配置](https://github.com/areal-project/AReaL/blob/v2.0.0/examples/math/gsm8k_grpo.yaml)
- 主实现采用 SGLang + FSDP2 + GRPO，BF16、gradient checkpointing、4K context、较低并发起步。
- 先做 1.5B smoke，再做 4B 系统验证；Harness controller 放 CPU。
- A100 40GB：7B/8B 优先 LoRA；A100 80GB：profile 后再决定 7B/8B 全参。
- 官方广泛测试参考硬件是 H800，不应把官方吞吐直接外推到 A100；实际容量以本机 profile 为准。[官方安装说明](https://areal-ai.io/AReaL/en/tutorial/installation.html)

### 6.2 规划估算

下表是排期预算，不是已测性能：

| 工作 | 预计 wall-clock | 8-GPU 小时 | 停止条件 |
|---|---:|---:|---|
| 官方 smoke + checkpoint/restore | 2–4 h | 16–32 | 基线不通则不改 Harness |
| 200–500 episode trace/profile | 4–8 h | 32–64 | log-prob/版本无法复算则修 schema |
| 单个核心实验臂 pilot | 8–24 h | 64–192 | reward 或环境 invalid rate 超门槛 |
| 4 臂 × 3 seeds 决定性实验 | 4–9 连续天 | 约 800–1600 | 以 pilot 实测重估后再批准 |

环境和 simulator 延迟可能使 wall-clock 更长而 GPU 利用率更低。运行前必须取得 A100 显存、P2P 拓扑、CPU/RAM、NVMe、Docker/网络边界。

### 6.3 临时共享 GPU 边界

当前每张 A100 只允许本项目新增使用最多 30 GiB，不能按 80 GiB 物理总显存配置进程。任何 GPU 命令启动前都必须重新读取 `memory.used` 与 `memory.free`；若其他进程已占约 50 GiB，则官方 8-GPU B0 和需要独占 GPU 的训练保持关闭。只有模型、KV cache、activation、CUDA Graph 与运行时余量的保守合计低于 30 GiB，且实际空闲显存仍留有安全余量时，才允许单卡 inference-only smoke。该临时边界不改变实验配置的长期目标，也不能通过降低检查阈值绕过。

## 7. 实施里程碑

| 阶段 | 预计人时 | 产物 | 硬退出条件 |
|---|---:|---|---|
| P0 基线与容量画像 | 2–3 天 | 固定 commit、官方 smoke、GPU/吞吐/恢复报告 | 官方基线与恢复不通 |
| P1 版本化数据面 | 3–5 天 | Registry、JointVersion、session pin、trace schema | 任一轨迹不能唯一还原版本 |
| P2 Harness-only 学习 | 3–5 天 | 冻结 policy 的 5-action controller、单元/恢复测试 | 人工两类任务不能学出不同动作偏好 |
| P3 同步联合更新 | 5–7 天 | 双 optimizer、barrier、复合 checkpoint、原子发布 | 1000 episode 中出现跨版本污染 |
| P4 慢速 artifact 演化 | 5–7 天 | bounded patch、shadow、paired canary、expire/rollback | 未验收候选进入 active set |
| P5 主实验与有界异步 | 1–2 周 | 2×2、3 seeds、lag 消融、tau2 held-out | 不满足预注册 go/no-go 则不扩到 SWE |

每阶段只在上一阶段退出条件通过后继续，避免在 trace 不可信时扩大算力。

## 8. 评测与归因

冻结旧/新模型与旧/新完整 Harness 快照

\[
H_j=(\omega_j,\phi_j,v_{tool,j},v_{parser,j},v_{context,j}),
\]

并在相同 environment 与冻结 evaluator \(E^*\) 下评测：

\[
M_{ij}=J(\theta_i,H_j;E^*),\quad i,j\in\{0,1\}.
\]

报告

\[
\Delta_{policy}=M_{10}-M_{00},\quad
\Delta_H=M_{01}-M_{00},
\]

\[
\Delta_{int}=M_{11}-M_{10}-M_{01}+M_{00}.
\]

所有组合匹配环境 episode、生成/上下文 token、工具与 verifier 调用、wall-clock、GPU-hours 和候选搜索数。主指标是 sealed task success；同时报告 historical regression、成本、延迟、invalid rate、Harness action entropy/KL、policy KL、两个 lag、discard rate 和 rollback。

项目级 go/no-go 门暂定：3 seeds 上 \(\Delta_{int}\) 的 task-bootstrap 95% 置信区间下界大于 0，且 \(M_{11}\ge\max(M_{10},M_{01})+3\) 个百分点；历史成功率相对旧系统下降不超过 2 个百分点。阈值只能在 pilot 后、解封 sealed test 前一次性冻结。

## 9. 主要风险与降级路线

| 风险 | 观测 | 首选处理 | 降级但仍可发表的结果 |
|---|---|---|---|
| 双重非平稳 | KL/entropy 抖动、interaction 跨 seed 变号 | 同步 lag=(0,0)、降低 controller lr、放慢 \(\phi\) | 研究 alternating 与 joint 的稳定边界 |
| 收益只是 test-time compute | 工具/token 增加后才涨分 | matched budget、Pareto 曲线、random action control | 报告成本条件化收益而非能力增益 |
| Harness reward hacking | 训练 reward 涨、sealed success 不涨 | 可执行 verifier、冻结 release gate、对抗单测 | 给出失败机制与防护基准 |
| 40GB OOM/吞吐过低 | optimizer/KV 峰值、GPU idle | 1.5B/4B、短 context、低并发、7B LoRA | 先发表系统/小模型因果结果 |
| tau2 simulator 主导噪声 | 同策略方差大、API 漂移 | 固定 simulator/seed/version，扩大 paired eval | 回到确定性工具环境验证方法 |

## 10. 代码工作包

建议在远程 AReaL fork 中按以下边界实现：

```text
jphrl/
├── harness/spec.py          # immutable spec / artifact
├── harness/registry.py      # pool, lineage, active set
├── harness/controller.py    # kappa_omega and optimizer
├── trajectory/schema.py     # joint event fields and migration
├── trajectory/areal_interaction_sidecar.py # model-call identity, tree and sample spans
├── trajectory/areal_agent_service_adapter.py # session/trajectory receipts and training record gate
├── trajectory/areal_data_proxy_pre_batch.py  # verified pre-batch callback contract
├── trajectory/areal_online_binding.py        # private staged/finalized online journal
├── trajectory/hermes_model_call_receipts.py  # Hermes per-upstream-call receipts
├── evolution/controller.py  # macro-step, barrier, validation, publish
├── checkpoint/manifest.py   # composite state and atomic pointer
├── eval/cross_play.py       # M00/M10/M01/M11
└── tests/                   # version, logprob, fault, replay, reward
```

先通过 adapter 接 AReaL workflow/proxy/trainer；只有无法暴露 batch barrier 时才提交一个小型上游 hook。不要把整个 AReaL trainer 复制成长期 fork。

## 11. 依赖的本地知识

- 统一综述：[[PaperNotes/Agentic-RL-Harness-统一范式|Agentic RL × Harness：近期工作、统一范式与差异]]
- 现代 RL 书籍：`Book/hands-on-modern-rl`，固定知识审计 commit `29e27088e01097ae6bd149313581a8ae5b68f65b`
- 详细实验协议：[EXPERIMENT_PLAN.md](refine-logs/EXPERIMENT_PLAN.md)
- 运行记录模板：[EXPERIMENT_TRACKER.md](refine-logs/EXPERIMENT_TRACKER.md)

## 12. 开工前唯一硬件门

先保存以下命令输出：

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi topo -m
nvidia-smi --query-gpu=driver_version --format=csv,noheader
lscpu
free -h
df -h / /tmp
```

在未确认 A100 40GB/80GB 与 GPU 拓扑前，不冻结 7B/8B、full/LoRA、rollout 并发或上下文长度。
