# JPH-RL 实验追踪表

状态只使用：`planned`、`running`、`passed`、`failed`、`stopped`、`invalid`。
`failed` 表示方法/任务失败；无法复算行为概率、mask 或版本的结果必须标 `invalid`。

## Frozen Context

| 字段 | 当前冻结值 |
|---|---|
| AReaL | `v2.0.0@fee938eada49208a5aabdbc1095730a13076a349` |
| 远端硬件 | 8×NVIDIA A100-SXM4-80GB |
| 远端根目录 | `/mnt/sdb/ljw/chizm`；禁止写入其外部 |
| 代码同步 | GitHub push/pull；模型、数据、环境与实验产物不入 Git |
| 起始 policy | 1.5B smoke；主模型待 pilot |
| 起始环境 | calculator contract → AReaL GSM8K B0 → tau2 |
| 首版 staleness | `(policy_lag, harness_lag)=(0,0)` |

## Run Ledger

| Run ID | Block | 目的/Arm | Seed | Policy | Harness | 状态 | 结果/证据 | 产物位置 | 备注 |
|---|---|---|---:|---|---|---|---|---|---|
| JPH-M0-LOCAL-001 | B0 | 本地 contract smoke | 0 | scripted mock | fixed smoke | passed | 29/29 tests；13-event trace | 本地 tests | 不构成真实模型/AReaL 结果 |
| JPH-M0-REMOTE-ENV-001 | B0 | SSH/GPU/磁盘预检 | 0 | N/A | N/A | passed | 8×A100 80GB；0/1/4–7 <500MiB；259GB 可用 | 会话日志，待写 profile | 系统 DNS 无上游；公网 IP 可达 |
| JPH-M0-GITHUB-001 | B0 | 目标根目录内 GitHub 登录 | 0 | N/A | N/A | running | device auth 已启动 | `/mnt/sdb/ljw/chizm/config/github` | 认证等待用户确认 |
| JPH-B0-OFFICIAL-001 | B0 | 官方 AReaL 1-step | 0 | base | fixed | planned |  | 外置 artifacts/logs | 必须先 clone、安装、路径审计 |
| JPH-B0-TRACE-001 | B0 | token/logprob/mask/version 复算 | 0 | base | fixed | planned |  | 外置 trace |  |
| JPH-B1-HO-S0 | B1 | Harness-only contextual bandit | 0 | frozen/not invoked | trainable | passed | 最差最优动作概率 0.9875；随机基线 0.2；参数 delta L2 累计 8.4126 | `/tmp/jphrl-harness-bandit.json`（待远端复跑） | 400 steps |
| JPH-B1-HO-S1 | B1 | Harness-only contextual bandit | 1 | frozen/not invoked | trainable | passed | 最差最优动作概率 0.9885；随机基线 0.2；参数 delta L2 累计 8.2224 | 同上 | 400 steps |
| JPH-B1-HO-S2 | B1 | Harness-only contextual bandit | 2 | frozen/not invoked | trainable | passed | 最差最优动作概率 0.9877；随机基线 0.2；参数 delta L2 累计 8.3015 | 同上 | 400 steps |
| JPH-B2-FX-S0 | B2/B3 | fixed | 0 | frozen | frozen | planned |  | 外置 run dir | pilot |
| JPH-B2-PO-S0 | B2/B3 | policy-only | 0 | trainable | frozen | planned |  | 外置 run dir | pilot |
| JPH-B2-HO-S0 | B2/B3 | Harness-only | 0 | frozen | trainable | planned |  | 外置 run dir | pilot |
| JPH-B2-JT-S0 | B2/B3 | joint | 0 | trainable | trainable | planned |  | 外置 run dir | pilot |

## Per-run 必填元数据

```yaml
run_id:
claim_ids: []
git:
  project_commit:
  areal_commit:
hardware:
  gpu_inventory_hash:
  topology_hash:
versions:
  policy_behavior:
  harness_controller_behavior:
  harness_artifact:
  tool_schema:
  parser:
  environment:
  training_evaluator:
  sealed_evaluator:
  tokenizer:
  context_builder:
budget:
  episodes:
  generated_tokens:
  context_tokens:
  tool_calls:
  verifier_calls:
  wall_clock_s:
  gpu_hours:
updates:
  policy_parameter_delta_norm:
  harness_parameter_delta_norm:
  policy_kl:
  harness_kl:
result:
  task_success:
  historical_regression:
  invalid_rate:
  mixed_version_episodes:
artifacts:
  resolved_config:
  joint_manifest:
  checkpoint:
  trajectory_journal:
  metrics:
decision:
  status:
  reason:
```

## Gate Checklist

### G0：允许进入 Harness-only

- [ ] 项目和 AReaL 均经 Git clone/pull 落到目标代码目录。
- [ ] 依赖、模型、缓存和 artifact 路径审计没有逃出目标根目录。
- [ ] 官方 8-GPU 1-step、actor update、weight sync 通过。
- [ ] 真实 token old log-prob、loss mask 和 policy version 可复算。

### G1：允许进入 joint pilot

- [x] 3 seeds Harness-only 均学出任务条件动作偏好（本地 sanity；远端复跑仅用于留存 artifact）。
- [ ] policy token 与 Harness action 的行为概率、mask、credit 分离。
- [ ] 1000 episode mixed-version=0。
- [ ] updater/publish 故障注入无半版本。
- [ ] checkpoint 恢复后下一 joint step 一致。

### G2：允许完整 3-seed 矩阵

- [ ] 单 seed 四臂预算匹配且 valid episode 达冻结门槛。
- [ ] 已完成 `M00/M10/M01/M11` cross-play。
- [ ] pilot 后 GPUh 已重估并获得用户批准。
- [ ] sealed test 尚未参与调参。

## Decision Log

| 日期 | 决策 | 证据 | 回滚方式 |
|---|---|---|---|
| 2026-08-02 | Hermes 只作为 policy 数据面参考，不视作 Harness learning | Hermes 捕获 LLM token/logprob 并训练 PPO，但无第二个 Harness optimizer/联合版本发布 | 保留上游固定 checkout，不修改 AReaL 主干 |
| 2026-08-02 | 先通过 B0/B1，再实现 P3 | 当前 25 tests 只覆盖固定 controller 与 trace contract | 阶段门失败即停，不消耗大规模 GPU |
| 2026-08-02 | 公网访问使用临时 SSH 反向 SOCKS，不修改服务器 DNS | 系统有公网路由但 systemd-resolved 无上游；代理下 TLS/ls-remote 成功 | 关闭 SSH tunnel 即完全移除 |

## Failure Log

| 时间 | Run ID | 症状 | 首个证据 | 分类 | 处理 |
|---|---|---|---|---|---|
| 2026-08-02 | JPH-M0-GITHUB-001 | `Could not resolve host` | `resolvectl status` 所有 link `Current Scopes: none` | infra | 使用不改系统状态的临时反向 SOCKS；认证仍需设备确认 |
