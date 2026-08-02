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
| JPH-M0-LOCAL-001 | B0 | 本地 contract smoke | 0 | scripted mock | fixed smoke | passed | 37/37 tests；13-event trace | 本地 tests | 不构成真实模型/AReaL 结果 |
| JPH-M0-REMOTE-ENV-001 | B0 | SSH/GPU/磁盘预检 | 0 | N/A | N/A | passed | 8×A100 80GB；2026-08-02 五次采样中最少空闲 77,843MiB；安装后约 230GiB 可用 | `/mnt/sdb/ljw/chizm/artifacts/bootstrap/` | GPU 0/2/3 有其他用户进程；不终止、不修改；系统 DNS 无上游 |
| JPH-M0-GITHUB-001 | B0 | 目标根目录内 GitHub 登录 | 0 | N/A | N/A | passed | `gh auth status`：账号 `CRhapsody`；项目经 push/pull 同步 | `/mnt/sdb/ljw/chizm/config/github` | Git 配置位于目标根目录的 runtime HOME |
| JPH-M0-AREAL-ENV-001 | B0 | 固定 AReaL 环境安装与 CUDA 校验 | 0 | N/A | N/A | passed | 472 个锁定包；PyTorch 2.9.1+cu129；FlashAttention CUDA 运算通过；路径审计 `ok=true` | `/mnt/sdb/ljw/chizm/artifacts/bootstrap/areal-v2.0.0/` | `v2.0.0@fee938e...`；tmux exit=0 |
| JPH-B0-PREFETCH-001 | B0 | 1.5B 模型与 GSM8K 预取 | 0 | base | fixed | passed | Qwen commit `989aa798...`；GSM8K commit `740312a...`，7473 train/1319 test | 外置 HF cache 与 bootstrap manifest | 约 3.0GiB；离线固定 snapshot |
| JPH-B0-MODEL-LOAD-001 | B0 | 固定模型离线 CUDA load/generate | 0 | base | fixed | passed | 1,543,714,304 参数；BF16；峰值 3,100,396,032 bytes；短生成成功 | `/mnt/sdb/ljw/chizm/artifacts/bootstrap/qwen2.5-1.5b-cuda-smoke.json` | GPU 0；不是 AReaL 训练结果 |
| JPH-B0-OFFICIAL-001 | B0 | 官方 AReaL 1-step | 0 | base | fixed | passed | run `20260802T072643Z` exit=0；16 seq、2540 valid train tokens、reward avg=0.625；`update_successful=1`；weight update 1.7268s；总训练 137.97s | `/mnt/sdb/ljw/chizm/artifacts/areal-b0/20260802T072643Z/` | 8×A100；每卡门禁 used≤10GiB 且 free≥70GiB；未终止或修改其他用户进程；只证明 policy 训练链路，不构成 Harness learning |
| JPH-B0-TRACE-001 | B0 | token/logprob/mask/version 复算 | 0 | base | fixed | passed | run `20260802T104241Z` exit=0；真实 AReaL rollout 1/1；103 prompt + 64 action tokens；六字段 roundtrip 最大误差 0；冻结 HF 快照复算 64/64 在预先固定容差内，mean/p95/max abs=`0.01627/0.11498/0.18852` | `/mnt/sdb/ljw/chizm/artifacts/areal-trace-b0/20260802T104241Z/` | GPU 0 峰值 28,958MiB；HF BF16 与 SGLang 存在数值差异；只证明行为数据可审计，不证明逐 token 位等同，也不构成 policy/Harness update |
| JPH-B1-HO-S0 | B1 | Harness-only contextual bandit | 0 | frozen/not invoked | trainable | passed | 最差最优动作概率 0.9875；随机基线 0.2；参数 delta L2 累计 8.4126 | `/mnt/sdb/ljw/chizm/artifacts/harness-bandit/b1-three-seed.json` | 远端复跑，400 steps |
| JPH-B1-HO-S1 | B1 | Harness-only contextual bandit | 1 | frozen/not invoked | trainable | passed | 最差最优动作概率 0.9885；随机基线 0.2；参数 delta L2 累计 8.2224 | 同上 | 远端复跑，400 steps |
| JPH-B1-HO-S2 | B1 | Harness-only contextual bandit | 2 | frozen/not invoked | trainable | passed | 最差最优动作概率 0.9877；随机基线 0.2；参数 delta L2 累计 8.3015 | 同上 | 远端复跑，400 steps |
| JPH-G1-CTRL-001 | G1/B4 | 联合控制面完整性 | 0 | toy candidate | toy candidate | passed | commit `9ccd14f...`；53/53 tests；8/8 更新干预、10/10 负向 mutation、6/6 发布故障场景通过；1000 个确定性版本 fixture 中 mixed/half/stale-accepted 均为 0；toy checkpoint 下一步一致 | `/mnt/sdb/ljw/chizm/artifacts/g1/20260802T125741Z/` | CPU-only synthetic fixture；不是 AReaL rollout、真实 optimizer update 或联合学习结果 |
| JPH-G1-BRIDGE-001 | G1/B0 | 真实 AReaL interaction 与 Harness sidecar | 0 | Qwen2.5-1.5B base | frozen tabular controller | invalid | commit `ab27606...`；真实 rollout、sidecar、prompt token 重算均为 4/4；冻结 HF 跨后端诊断仅 3/4 通过，且落盘配置含默认公开 admin key；正式 `audit.json` 未生成 | `/mnt/sdb/ljw/chizm/artifacts/areal-joint-bridge/20260802T131531Z/` | GPU 0；tmux exit=1；失败产物原样保留；无 policy/Harness optimizer update |
| JPH-G1-BRIDGE-002 | G1/B0 | 同后端行为概率正式复算 | 0 | Qwen2.5-1.5B base | frozen tabular controller | invalid | commit `30d8346...`；4/4 真实 rollout、同一 JointVersion 与 prompt 干预均已落盘；AReaL v2.0.0 转换完整 SGLang score 数组时遇到 `None`，而已安装 SGLang 的 regular-request 契约明确含前缀 `None`；score=0、`audit.json` 不存在；3 个配置字段仍含默认管理 key | `/mnt/sdb/ljw/chizm/artifacts/areal-joint-bridge/20260802T134056Z/` | GPU 0 峰值 28,958MiB；tmux exit=1；measurement/security invalid；失败产物原样保留；无 optimizer update |
| JPH-G1-BRIDGE-003 | G1/B0 | SGLang score compatibility adapter | 0 | Qwen2.5-1.5B base | frozen tabular controller | invalid | commit `1b3a379...`；worker 确认实例化项目 adapter，4/4 rollout 后 `controller.compute_logp` 已返回；score provenance 序列化遇到 AReaL `RTensor`，score=0、`audit.json` 不存在 | `/mnt/sdb/ljw/chizm/artifacts/areal-joint-bridge/20260802T135147Z/` | GPU 0 峰值 28,956MiB；tmux exit=1；敏感、权限、路径审计通过；无 optimizer update |
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

- [x] 项目和 AReaL 均经 Git clone/pull 落到目标代码目录。
- [x] 已安装依赖、缓存和 artifact 的路径审计没有逃出目标根目录；模型/数据预取完成后再复审。
- [x] 官方 8-GPU 1-step、actor update、weight sync 通过。
- [x] 真实 token old log-prob、loss mask 和 policy version 可复算；64/64 个 action token 在预先固定容差内通过，只证明行为数据可审计，不证明 HF 与 SGLang 逐 token 位等同。

### G1：允许进入 joint pilot

- [x] 3 seeds Harness-only 均学出任务条件动作偏好（本地 sanity；远端复跑仅用于留存 artifact）。
- [x] synthetic CPU fixture 中 policy token 与 Harness action 的行为概率、mask、credit 和 toy updater 依赖已分离。
- [x] synthetic 本地 POSIX 控制面中 updater/publish 故障注入无半版本，toy checkpoint 恢复后下一步一致。
- [ ] 真实 AReaL interaction 已绑定生效的 Harness decision sidecar 与同一 `JointVersion`。
- [ ] 1000 条真实 episode mixed-version=0。
- [ ] 真实 policy/Harness optimizer 与生产 checkpoint 的故障注入、整对发布和恢复通过。

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
| 2026-08-02 | 公网访问使用目标目录内的 loopback CONNECT 代理，不修改服务器 DNS | 系统有公网路由但 systemd-resolved 无上游；显式 DNS、TLS、GitHub 与依赖安装均通过 | 停止 `jph-net` tmux session 即完全移除 |
| 2026-08-02 | 所有长任务使用目标目录内的显式 tmux socket | 默认 tmux socket 会写 `/tmp`；`/mnt/sdb/ljw/chizm/runtime/tmux/jph.sock` 在 SSH 重连后仍存活 | 停止相应 session；日志与 socket 均在目标根目录 |
| 2026-08-02 | B0 以显存余量代替 `<500MiB` 空闲判定 | SGLang `mem_fraction_static=0.8` 对 80GiB 卡静态预算为 64GiB；70GiB 最小空闲仍留 6GiB；actor 调度声明 32GiB | 默认要求 used≤10GiB 且 free≥70GiB；环境变量可收紧门禁，启动脚本再次检查 |
| 2026-08-02 | B0 非回环 proxy 使用每次 run 独立的随机 admin key | AReaL 2.0.0 会拒绝非回环地址上的默认 key；run `20260802T072643Z` 的 proxy 与 proxy-eval 均成功初始化 | key 只经环境变量传入，argv 保留 OmegaConf 环境引用；产物权限 0600，配置写出后立即脱敏并在 EXIT 再审计 |
| 2026-08-02 | G0 行为轨迹审计门通过，但不把跨后端容差通过写成精确复算 | run `20260802T104241Z` 的 AReaL `ModelResponse` 到正式六字段张量 roundtrip 最大误差为 0；冻结 HF BF16 前向对 64 个 action token 的 mean/p95/max abs 为 `0.01627/0.11498/0.18852`，全部低于 run 前已提交的 mean≤0.05、max≤0.25 门槛 | 若后续训练对 importance ratio 偏差更敏感，先冻结同后端重算或更严格阈值，再进入 joint pilot |
| 2026-08-02 | G1 synthetic 控制面证据通过，但 joint pilot 门继续关闭 | `JPH-G1-CTRL-001` 证明两路 toy updater 的干预不变性、lag0 gate、原子 pair publish 与 toy replay；结果文件明确 `areal_policy_update=false`、`production_harness_update=false` | 下一步先做单卡真实 AReaL interaction bridge；不得把 1000 个 deterministic fixture 写成真实 rollout |
| 2026-08-02 | 真实 bridge 的正式概率门改为同一 AReaL/SGLang 后端复算，冻结 HF 只保留诊断 | `JPH-G1-BRIDGE-001` 中 4 条真实轨迹均可生成，但 HF BF16 与 SGLang 在一条轨迹出现 importance-ratio 尾部差异；事后把旧 max 阈值从 0.25 放宽到 0.28 会污染预注册 | 新 run 预先固定每条轨迹 mean importance-ratio error≤0.02、max≤0.10；HF 仍完整报告但不阻断；任一同后端轨迹失败则不生成通过审计 |

## Failure Log

| 时间 | Run ID | 症状 | 首个证据 | 分类 | 处理 |
|---|---|---|---|---|---|
| 2026-08-02 | JPH-M0-GITHUB-001 | `Could not resolve host` | `resolvectl status` 所有 link `Current Scopes: none` | infra | 使用不改系统状态的临时反向 SOCKS；认证仍需设备确认 |
| 2026-08-02 | JPH-B0-OFFICIAL-001 | `UV_MANAGED_PYTHON` cannot be used with `--python-preference` | uv 0.11.26 在 Python 下载前拒绝互斥环境开关 | infra/config | 删除冗余 `UV_MANAGED_PYTHON=1`，保留 `UV_PYTHON_PREFERENCE=only-managed`；加回归测试后从同一 bootstrap 继续 |
| 2026-08-02 | JPH-B0-OFFICIAL-001 | lockfile needs update under `--locked` | 脚本用清华 `--default-index` 改变了官方 lock 中的来源 URL；去掉覆盖后 `uv lock --check` 1ms 通过 | infra/config | AReaL exact sync 保留官方 index/哈希并走 HTTPS 代理；普通 pip smoke 仍可用清华镜像 |
| 2026-08-02 | JPH-B0-OFFICIAL-001 | `nvidia-nccl-cu12` 下载重试后超时 | SSH 反向 HTTP 隧道随交互连接中断；其余约 8.8GB uv cache 已保留 | infra/network | 代理与 bootstrap 分别放入目标根目录 socket 的 tmux session；代理只监听 loopback 且只允许 CONNECT 443 |
| 2026-08-02 | JPH-B0-PREFETCH-001 | GSM8K 一次性 `python -c` 立即报 `NameError` | 多层 SSH/tmux/shell 引号剥离了数据集字符串的引号 | orchestration | 改为仓库内的可测试脚本，固定 dataset commit 并审计 materialized cache path |
| 2026-08-02 | JPH-B0-OFFICIAL-001 | 显存门禁通过后立即 `Permission denied`，tmux exit=126 | `run_areal_official_b0.sh` 在 Git 中是 100644，waiter 直接将它当可执行文件调用 | orchestration/config | waiter 改用 `/bin/bash` 显式解释脚本；训练尚未启动，因此没有残留 GPU 进程 |
| 2026-08-02 | JPH-B0-OFFICIAL-001 | Hydra 拒绝 `total_train_steps=1`，tmux exit=1 | 固定 YAML 没有该键，虽然 `GRPOConfig` 定义了字段；Hydra struct 模式要求 `+total_train_steps=1` | config | 用 `+` 显式追加字段；失败发生在 worker 启动前，只有 GPU 监控日志 |
| 2026-08-02 | JPH-B0-OFFICIAL-001 | AReaL worker 命令把编译缓存指向 `/tmp/areal-ljw` | 上游 launcher 在未设置 `AREAL_CACHE_DIR` 时使用用户级 `/tmp` 默认值 | path policy/config | 立即停止本次 run；在 `remote_env.sh` 固定 `AREAL_CACHE_DIR=${JPH_ROOT}/cache/areal`，重启后检查 worker 命令；`/tmp` 下只留下空目录，因无目录外写权限不擅自删除 |
| 2026-08-02 | JPH-B0-OFFICIAL-001 | run `20260802T070510Z` 的四个 SGLang 子进程均报 `runpy`/`NamespaceLoader` ImportError | AReaL 用裸 `python3` 启动 SGLang；PATH 命中系统 Python，但继承的 `PYTHONPATH` 指向固定 3.12 标准库 | runtime/config | 停止本次 run；把 `${AREAL_VENV}/bin` 放在 B0 的 PATH 首位，确保所有子进程使用同一解释器 |
| 2026-08-02 | JPH-B0-OFFICIAL-001 | run `20260802T070837Z` 的 SGLang fused-rope JIT 被 nvcc 拒绝：`Value 'c++20' is not defined` | 默认 `/usr/local/cuda` 指向 CUDA 11.8；PyTorch runtime 为 CUDA 12.9，SGLang JIT 需要 C++20 | runtime/config | 停止本次 run；使用服务器已有的只读 `/usr/local/cuda-12.6` 编译器，并在每次启动前做 C++20 编译检查；不下载新工具链 |
| 2026-08-02 | JPH-B0-OFFICIAL-001 | run `20260802T071439Z` 在 proxy-rollout 初始化时拒绝默认 admin key，tmux exit=1 | 非回环 host `10.103.9.44` 使用默认 `areal-admin-key`，触发 AReaL 安全检查；上游随后清理全部自有 worker | runtime/security | 每次 run 生成独立随机 key；不使用 `AREAL_ALLOW_DEFAULT_ADMIN_KEY=1` 绕过；增加权限收紧、落盘脱敏和路径约束测试 |
| 2026-08-02 | JPH-B0-OFFICIAL-001 | run `20260802T072520Z` 在 Hydra 合成阶段拒绝 admin key override，tmux exit=1 | 固定 YAML 没有 `rollout.agent.admin_api_key`，struct 模式要求 `+rollout.agent.admin_api_key=...` | config | 用 `+` 显式追加；失败发生在 worker 启动前；run `20260802T072643Z` 随后完整通过 |
| 2026-08-02 | JPH-B0-TRACE-001 / `20260802T103007Z` | trace 配置合成失败 | `rollout.agent=null` 不能赋给非可选的 agent 配置 | config/invalid | 保留结构化默认 agent 配置；没有 rollout 或 trace |
| 2026-08-02 | JPH-B0-TRACE-001 / `20260802T103236Z` | 单任务 trace runner 在数据控制器初始化前失败 | 上游 `RDataset` 尚未连接 `DataController` | runtime/invalid | 直接调用上游公开 `get_custom_dataset` 读取固定 GSM8K snapshot；没有 rollout 或 trace |
| 2026-08-02 | JPH-B0-TRACE-001 / `20260802T103414Z` | SGLang 子进程报 `runpy`/`NamespaceLoader` ImportError | worker 的裸 `python3` 命中系统解释器 | runtime/invalid | trace launcher 将固定 AReaL venv 放到 PATH 首位；仅清理本次自有进程，没有 trace |
| 2026-08-02 | JPH-B0-TRACE-001 / `20260802T103626Z` | SGLang CUDA graph JIT 拒绝 C++20 | 默认 nvcc 来自 CUDA 11.8 | runtime/invalid | 固定只读 CUDA 12.6，并在启动前执行 C++20 preflight；仅清理本次自有进程，没有 trace |
| 2026-08-02 | JPH-B0-TRACE-001 / `20260802T103913Z` | 真实生成后 interaction 构造失败 | 固定 AReaL 提交的 `InteractionWithTokenLogpReward` 不定义 `original_reward` | schema/invalid | 删除错配字段，严格采用上游六字段 `to_tensor_dict()` 契约；AReaL 自行回收 worker，没有 trace |
| 2026-08-02 | JPH-G1-CTRL-001 preflight | 下发的完整 commit hash 不存在 | 错误字符串与真实提交只共享短前缀；远端完整对象 ID 门禁拒绝继续 | orchestration/stopped | 不创建 tmux、不运行实验；之后只使用本地 `git rev-parse HEAD` 的 40 位结果 |
| 2026-08-02 | JPH-G1-CTRL-001 preflight | launcher 默认解释器路径不存在 | 脚本写成 `areal-v2`，服务器固定环境为 `areal-v2.0.0`；路径门禁在 Git pull 前停止 | config/stopped | 修正默认路径并增加回归测试；正式 run 验证未设置 `JPH_PYTHON` 时仍可运行 |
| 2026-08-02 | JPH-G1-BRIDGE-001 / `20260802T131531Z` | 4 条真实 rollout 完成后，正式 bridge 审计失败 | 第 4 条轨迹 HF/SGLang max abs logprob error=`0.277804`，超过预先固定的 `0.25`，对应 importance-ratio error=`0.242555`，因此仅 3/4 通过；同时四份 AReaL 配置含默认公开值 `areal-admin-key` | measurement/security/invalid | 保留失败 artifact；不放宽旧阈值。后续用官方 `controller.compute_logp` 做同后端正式复算，阈值在重跑前提交；每次生成随机 admin key，经环境变量传入并在审计前对整棵 run tree 脱敏 |
| 2026-08-02 | JPH-G1-BRIDGE-002 / `20260802T134056Z` | 4 条 rollout 后同后端 score API 在落盘前失败，且敏感门仍不通过 | AReaL `SGLangBackend.parse_score_response` 先对完整 `input_token_logprobs` 执行 `float(x[0])`；SGLang 0.5.10.post1 regular request 明确返回 `[None] + input_token_logprobs[:-1]`，触发 `NoneType`；另有 3 个默认管理 key 字段 | compatibility/security/invalid | 新增薄 adapter：严格先取官方定义的最终 `target_len` score tail，再转换；tail 内 `None`、短返回、非有限值继续拒绝。脱敏器同时清除运行时随机 key 与公开默认 key；不修改 AReaL 固定 checkout，不放宽 2%/10% 门槛 |
| 2026-08-02 | JPH-G1-BRIDGE-003 / `20260802T135147Z` | 同后端 score 已计算，但 provenance 写入前无法序列化 trajectory | `controller.compute_logp(results)` 已返回；AReaL controller 的 trajectory 仍含延迟获取的 `RTensor`，项目 `_plain()` 不能将它直接交给 JSON | compatibility/invalid | 在 controller 销毁前使用 AReaL 官方 `RTensor.localize` 批量本地化 rollout 与 rescored tensors；schema v4 显式记录 transport localization；不使用不透明 duck typing，不改变 token、mask 或 logprob |
