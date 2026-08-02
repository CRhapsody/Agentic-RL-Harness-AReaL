# JPH-RL 实验计划

**问题**：怎样在 AReaL 的真实 Agent rollout 数据面上，让模型 policy
`pi_theta` 与具有行为概率的 Harness controller `kappa_omega` 从同一冻结批次
得到可区分的更新，并把两者作为一个联合版本发布？

**方法论主张**：一次联合宏步固定
`V_k=(theta_k, omega_k, artifact_k, tool/parser/env/eval/tokenizer/context versions)`；
同一批反馈分别产生 policy 与 Harness 候选更新，只有两个更新及独立验收全部通过
时才原子发布 `V_{k+1}`。

**日期**：2026-08-02

## Claim Map

| Claim | 为什么重要 | 最低可信证据 | 实验块 |
|---|---|---|---|
| C1：policy 与 Harness 均发生可区分的非零学习更新，并形成正交互 | 完整系统变好不等于两部分都学会了 | 两类 old log-prob/credit 可复算；参数 delta 均非零；2×2 cross-play 中 `Delta_interaction > 0` | B0–B3 |
| C2：联合版本、二维 staleness 和整对发布保证闭环可恢复 | 两个独立更新器会引入半版本和分布错配 | 1000 episode 无混版本；故障注入无半发布；恢复后下一宏步一致 | B0、B4 |

需要排除的解释：收益仅来自更多 token、工具调用、重试、搜索候选或 wall-clock；
Harness 只是固定 prompt；同一个终局 reward 被无条件复制成两路 credit；训练 evaluator
漂移制造了虚假进步。

## Paper Storyline

- 主文必须证明：真实 AReaL policy 基线可复现；Harness-only 可学习；联合更新优于
  fixed、policy-only 和 Harness-only；收益在等预算与冻结 evaluator 下仍存在。
- 附录支持：二维 lag、checkpoint/rollback、action-disable-one、skill 删除/随机替换。
- 暂时删除：SWE/Terminal/Web、任意 Python 自修改、第二个大模型 Harness writer。

## Experiment Blocks

### B0：真实数据面与版本正确性

- **Claim tested**：C1/C2 的测量前提。
- **任务**：AReaL v2.0.0 官方 GSM8K 1-step smoke；确定性 calculator trace。
- **系统**：官方 baseline；JPH-RL mock/HF contract smoke。
- **指标**：六字段训练 tensor roundtrip、token old log-prob、loss mask、policy version、
  episode 联合版本、峰值显存、无效轨迹率。
- **设置**：固定 AReaL commit `fee938eada49208a5aabdbc1095730a13076a349`；
  8×A100 80GB；AReaL 产物全部放仓库外。
- **成功条件**：官方 actor 完成 1 次更新和权重同步；抽样 token 数据可复算；无混版本。
- **失败解释**：数据面不可信，停止 Harness 改造，不产生科学结论。
- **优先级**：MUST-RUN。

### B1：Harness-only 可学习性

- **Claim tested**：Harness 不是固定规则，而是有参数、有行为概率、有独立 credit 的策略。
- **任务**：两个上下文 bandit；两个任务域的最优 Harness action 不同。
- **系统**：均匀随机、固定规则、trainable categorical controller。
- **指标**：每域正确动作概率、平均回报、entropy、KL、参数 delta、old log-prob 复算误差。
- **设置**：冻结模型；3 seeds；相同 action budget；先用纯 CPU 快速闭环。
- **成功条件**：每域最优动作概率均超过 0.8，且 held-out 平均回报显著高于随机策略。
- **失败解释**：状态、action 或 credit 定义无可学习信号，不进入联合训练。
- **优先级**：MUST-RUN。

### B2：同步联合更新最小闭环

- **Claim tested**：C1。
- **任务**：确定性 tool-use 任务族，之后才迁移 tau2 airline/retail。
- **系统**：fixed、policy-only、Harness-only、joint 四臂。
- **指标**：`||Delta theta||`、`||Delta omega||`、两路 KL/entropy、两路 advantage、
  task success、invalid rate 和预算。
- **设置**：所有 episode 固定同一 `V_k`；同一 batch 分离 credit；lag=(0,0)；
  任一 updater 或 release gate 失败则整对拒绝。
- **成功条件**：两个候选参数 delta 非零；下一轮 rollout 同时受新 policy 与新 Harness
  影响；不存在单边 active version。
- **失败解释**：若只有一边非零，只能称 policy-only 或 Harness-only；若两边变化但无法
  cross-play 归因，只能称 system co-adaptation，不能声称 synergy。
- **优先级**：MUST-RUN。

### B3：2×2 交叉归因与等预算比较

- **Claim tested**：C1 的因果归因。
- **任务**：固定 tau2 版本、互斥 train/validation/regression/sealed split。
- **系统**：`M00=J(theta0,H0)`、`M10=J(theta1,H0)`、`M01=J(theta0,H1)`、
  `M11=J(theta1,H1)`；另加 matched-compute random controller。
- **主指标**：sealed task success；次指标为 historical regression、token、工具调用、
  wall-clock、GPUh、candidate count。
- **成功条件**：3 seeds 的 task-bootstrap 95% CI 下界满足
  `Delta_interaction=M11-M10-M01+M00 > 0`，且 `M11` 至少比 `max(M10,M01)`
  高 3 个百分点；历史成功率下降不超过 2 个百分点。阈值只允许在 pilot 后、sealed
  test 解封前修改一次。
- **优先级**：MUST-RUN。

### B4：恢复、二维 staleness 与失败诊断

- **Claim tested**：C2。
- **比较**：lag=(0,0)/(1,1)；仅 policy lag/二维 lag；同步/交替；完整/缺失整对回滚。
- **指标**：mixed-version episode、stale discard、effective sample size、吞吐、恢复后
  batch/action probability/optimizer state 是否一致。
- **成功条件**：故障注入不产生半版本；恢复可重复；放宽 lag 的吞吐收益不以统计偏差
  为代价。
- **优先级**：C2 的 MUST-RUN，artifact 演化为 NICE-TO-HAVE。

## Run Order and Milestones

| Milestone | 目标 | 首批 Run | Go/Stop Gate | 估计成本 | 主要风险 |
|---|---|---|---|---:|---|
| M0 | 环境与 contract sanity | JPH-M0-LOCAL、JPH-B0-OFFICIAL | 真实 AReaL 1-step 和 trace audit 全过 | 16–32 GPUh | DNS、磁盘、依赖 ABI |
| M1 | Harness-only | JPH-B1-HO-S0..2 | 每域最优动作概率 >0.8 | <1 CPUh | reward/状态无信息 |
| M2 | joint toy/工具任务 | JPH-B2-JT-S0 | 双参数 delta、原子发布、无混版本 | 8–32 GPUh | 双重非平稳 |
| M3 | tau2 单 seed pilot | 四臂 seed 0 | valid episode >=95%，预算匹配 | 128–256 GPUh | simulator 噪声 |
| M4 | 3-seed 决定性实验 | 四臂 seeds 0–2 | pilot 后重新估算并批准 | 累计 800–1600 GPUh | 交互项跨 seed 变号 |

## Compute and Data Budget

- 远端实测为 8×A100-SXM4-80GB；当前数据盘可用约 259GB，已经是硬约束。
- 只缓存一个 1.5B smoke 模型和一个固定 AReaL 环境；下载前记录预计大小，下载后记录
  实际大小。7B/8B 与 tau2 数据必须等 B0/B1 后再批准。
- 代码目录：`/mnt/sdb/ljw/chizm/src/`；模型、数据、环境、缓存、checkpoint、trace、
  日志分别位于目标根目录的同名外置目录，禁止进入 Git 仓库。
- 完整 3-seed 矩阵不是本轮自动启动项；先完成 M0/M1。

## Risks and Mitigations

- **远端 DNS 无上游**：不修改系统文件；当前会话用 SSH 反向 SOCKS 访问公网，并把网络
  预检结果记为实验元数据。
- **磁盘 97% 已用**：单模型、单环境、下载前后审计；任何额外大文件先做预算。
- **收益来自额外推理计算**：逐 episode 记录 token/tool/verifier/wall-clock，加入等预算随机
  controller。
- **reward hacking/evaluator drift**：训练 evaluator 与 sealed evaluator 分离，release gate
  不可由训练分支修改。
- **AReaL fork 长期漂移**：目标仓库只放 adapter；AReaL 使用固定上游 checkout，不复制
  trainer 主干。

## Final Checklist

- [x] 核心 claim、反命题和退出条件已冻结
- [x] 主实验与 nice-to-have 已分离
- [ ] 官方 AReaL B0 通过
- [ ] Harness-only 可学习性通过
- [ ] 同一 batch 的双候选更新与原子发布通过
- [ ] 2×2 cross-play 和 matched-budget 完成
