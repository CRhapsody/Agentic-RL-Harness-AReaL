# JPH-RL

JPH-RL 是 `PROJECT_PLAN.md` 的增量可执行实现。当前代码已经覆盖 calculator B0 控制流、真实 AReaL interaction 身份、Hermes/DataProxy 在线接合，以及 Q/R/S 的双路训练样本准入和冻结 credit 对齐；真实 Policy 与 Harness optimizer、候选 checkpoint 和联合发布仍未接入。

## 本地无依赖检查

```bash
python3 -m unittest discover -s tests -v
python3 -m jphrl.cli --backend mock --task add-17-25 --output /tmp/jphrl-smoke.json
```

`mock` 后端只检查数据面与控制流，不构成模型能力结果。

## 远程真实模型预检

远程目录固定为 `/mnt/sdb/ljw/chizm`。同步代码后执行：

```bash
cd /mnt/sdb/ljw/chizm/src/Agentic-RL-Harness-AReaL
source scripts/remote_env.sh
bash scripts/bootstrap_remote.sh
source /mnt/sdb/ljw/chizm/venvs/jphrl-smoke/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m jphrl.cli \
  --backend hf \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda \
  --task add-17-25 \
  --output /mnt/sdb/ljw/chizm/artifacts/smoke/qwen25-05b-add.json
```

GPU 编号必须在运行前根据 `nvidia-smi` 选择；不得假设 GPU 0 空闲。

这条 HF 命令只验证一张空闲 GPU 上的真实生成 token、old log-prob 与 loss mask，**不算 AReaL B0**。AReaL B0 还必须经过官方 SGLang rollout、每-token `versions`、六字段训练 tensor roundtrip、FSDP actor 更新和权重同步。准备好固定 commit 的 AReaL 环境后，使用 `scripts/run_areal_official_b0.sh` 执行 1-step、8-GPU 官方 GSM8K smoke。

从干净服务器建立原生 AReaL 环境的入口是：

```bash
cd /mnt/sdb/ljw/chizm/src/Agentic-RL-Harness-AReaL
source scripts/remote_env.sh
bash scripts/bootstrap_areal_v2.sh
bash scripts/run_areal_official_b0.sh
```

bootstrap 会先检查 Hugging Face 镜像、清华 PyPI、GitHub、uv、Flash Attention、PyTorch 和 Python artifact 端点，再用官方版本化 standalone 安装器把 `uv==0.11.26` 直接安装到目标根目录；它不依赖系统 `python3-venv`、pip 或 Conda。随后下载受管 Python 3.12、创建独立 AReaL venv、clone 并核验固定 commit，再按 `uv.lock` 执行 `uv sync --locked --extra cuda`；`--locked` 会拒绝与 `pyproject.toml` 不一致的 lock，而不会静默改写。`0.11.26` 是已发布官方 SGLang 镜像中的实际 uv 版本；源码本身没有固定 uv CLI。由于 Flash Attention 不在 v2.0.0 的 exact lock 中，脚本会在 sync 后用 `--no-deps` 安装官方预编译 wheel，防止它被 exact sync 删除；最后在一张低于 500 MiB 占用的 GPU 上运行官方安装验证和 Flash Attention CUDA kernel smoke、冻结依赖清单并执行路径审计。

AReaL 固定为 `v2.0.0@fee938eada49208a5aabdbc1095730a13076a349`，要求 Python `>=3.11,<3.13`。官方 SGLang 镜像的 amd64 digest 冻结为 `sha256:2c6cc290a04139deb94400db74274f1f106f59d018a0df3ec19d76f154574147`。官方容器虽然最省依赖冲突，但 Docker 镜像层默认写到 daemon 的 `/var/lib/docker`；在“所有依赖都位于 `/mnt/sdb/ljw/chizm`”的硬约束下，只有远程 `docker info` 证明 `DockerRootDir` 已位于目标根目录时才可使用，否则改用目标根目录内的 uv/venv 环境。

## AReaL interaction 身份与训练样本归档

`jphrl/trajectory/areal_interaction_sidecar.py` 提供两层显式契约：

1. `build_interaction_adapter_sidecar()` 将本项目的 `model_call_id` 与 AReaL 的 `interaction_id` 一一绑定，并保存 episode、session、trajectory、parent、顺序与 `JointVersion`。
2. `export_bound_training_sample_archive()` 调用 AReaL 原生 `InteractionCache.export_interactions()`，支持 `individual` 与 `concat`，同时核验每次模型动作在六字段训练张量中的 token 区间。

归档器只证明“训练样本怎样形成以及属于哪次模型调用”，不会伪造 optimizer update 证据。当前 RLVR bridge 已使用单 interaction sidecar；多轮 Agent Service 的 N/O/P 接线和 Q/R/S 的样本/credit 准入均已实现。它们证明 receipt、轨迹、pre-batch 样本、行为时概率与冻结 advantage 的身份闭合，仍不构成 policy 或 Harness optimizer 已更新的证据。

`jphrl/trajectory/areal_agent_service_adapter.py` 进一步实现了多轮 Agent Service 接线契约：从 `rl/start_session` 提取不含凭据的 session receipt，从 OpenAI completion/response ID 捕获 interaction receipt，从 `rl/set_reward` 提取 ready trajectory receipt，并在 `EpisodeTrace`、session、trajectory、parent 树与 token 张量全部一致后生成训练记录。正确 hook 位于 AReaL `SessionData.export_trajectory()` 之后、`concat_padded_tensors()` 之前；公开 `/export_trajectories` 响应已经丢失逐 interaction 身份，不能事后用 batch 行号猜测绑定。

N/O/P 的生产边界由以下三个模块组成：

1. `jphrl.hermes_agent_service.HermesAgent` 是固定 AReaL Hermes 示例的显式子类入口。它锁定 `hermes-agent==0.19.0`，为每次真实上游响应暴露精确五字段 receipt：`model_call_id`、`interaction_id`、`ordinal`、`parent_model_call_id`、`session_id`。凭据字段和额外 metadata 不能进入 receipt。
2. `patches/areal-v2.0.0-data-proxy-pre-batch-hook.patch` 给固定 AReaL 增加最薄 callback。部署时设置 `AREAL_PRE_BATCH_HOOK=jphrl.trajectory.areal_online_binding.pre_batch_bind_agent_service_training_record`；callback 在每条 trajectory export 后、merge/tensorize 前执行，异常直接终止导出。
3. `stage_agent_service_training_binding()` 先把 Hermes receipts、完整 `EpisodeTrace`、session/trajectory receipts 与 `JointVersion` 写入项目外的私有 journal。pre-batch callback 再用真实 interaction mapping 调用既有 `prepare_agent_service_training_record()`，并 exactly-once 写入 training record 和 finalized marker。marker 明确保持 `policy_optimizer_update=false`、`harness_optimizer_update=false`。

Hermes 运行依赖单独固定在 `requirements-hermes.txt`。self-evolution caller 必须把 `rl/start_session` 返回的非秘密 `session_id` 放入 `metadata.jphrl_inference_session_id`；`session_api_key` 只用于路由，不能代替 session identity，也不会持久化。

固定 AReaL v2.0.0 的 CPU 集成验证入口是：

```bash
source scripts/remote_env.sh
/mnt/sdb/ljw/chizm/venvs/areal-v2.0.0/bin/python \
  scripts/verify_areal_agent_service_adapter.py
```

该验证沿完整 N/O/P journal + hook 路径覆盖真实 `SessionData` 的 `individual` 两样本与 `concat` 单叶样本，不初始化 CUDA，也不执行 optimizer。

## Q/R/S：真实样本准入与冻结双路 credit

N/O/P finalized training record 之后有三个独立的 fail-closed 边界：

1. `areal_policy_admission.py` 重验 P record、interaction archive 和 lag-zero `JointVersion`，保留 AReaL 六字段张量以及每个 `model_call_id` 的精确 decision span。一个 span 内只能有一个真实 inference engine version；trainable old log-prob 必须有限且非正。
2. `harness_action_admission.py` 从同一 `EpisodeTrace` 和同一 P record 准入真实 Harness decision，保存完整 `HarnessState`、固定五动作 schema、action mask、mask 前 logits、可重算 old log-prob、loss mask 和 behavior version。`infrastructure_invalid` 与 `trace_contract_invalid` episode 不能进入该边界。
3. `joint_credit_alignment.py` 把持久 Q/R admission 固定到同一 episode、P record 与 `JointVersion`。当前显式 estimator 是“terminal return 减去两份冻结 baseline”；Policy 与 Harness 的 source、baseline snapshot 和 target map 必须分开，synthetic/placeholder provenance 会被拒绝。Policy advantage 只写入 decision span 且严格等于 loss mask；Harness masked advantage 只受自己的 loss mask 控制。

S record 内嵌完整 Q/R admission 并保存 canonical SHA-256，因此 JSON 落盘后会重新验证来源记录、目标、版本、张量、mask 和 advantage，而不是只相信散列标签。`individual` 与 `concat` 都经过同一验证入口：

```python
policy = build_policy_training_admission(p_record, active_joint_version=version)
harness = admit_real_harness_action_samples(
    trace=trace,
    active_joint_version=version,
    pre_batch_training_record=p_record,
)
credit_record = build_frozen_joint_credit_alignment(
    policy_admission=policy,
    harness_admission=harness.to_record(),
    active_joint_version=version,
    estimator=frozen_dual_baseline,
)
```

上述入口只构造 optimizer-ready 的受审数据对象，不调用 optimizer。所有 Q/R/S evidence 仍固定为 `policy_optimizer_update=false`、`harness_optimizer_update=false`；后续 T/U 才能产生真实参数更新证据。

## 成功条件

一次 smoke 只有同时满足以下条件才通过：

1. episode 内所有事件绑定同一个 `JointVersion`；
2. Harness 先选择 `DIRECT` 发起模型/工具链路，再选择 `VERIFY` 检查工具结果；两个动作均记录行为策略旧对数概率、动作掩码和 mask 前 logits；
3. 模型生成合法 calculator 调用；
4. calculator 的执行结果等于任务答案；
5. 模型最终答案等于工具结果；
6. verifier 返回 `passed=true`；
7. 成功轨迹严格包含 13 个有序事件，并由 `event_id`/`parent_event_id` 串联；
8. 真实模型响应必须携带等长且非空的 completion token IDs、old log-probs 与 loss mask；scripted mock 明确标记 `not_applicable`，不能冒充真实 token 数据面；
9. 完整 trace 写入 `/mnt/sdb/ljw/chizm/artifacts/`。

成功 episode 的冻结预算是 **2 次模型调用、1 次工具调用、1 次 verifier 调用**。重试轨迹仍会被完整记录，但即使最终答案正确，只要模型调用数超过 2，也按 `policy_failure` 记 0 分，不能静默放宽 smoke 门槛。

## 目录约束

`scripts/remote_env.sh` 会把 `HOME`、Hugging Face、PyTorch、pip、Triton、CUDA、XDG、临时目录和日志全部重定向到 `/mnt/sdb/ljw/chizm`，并限制 CPU/编译并发。脚本拒绝在目标根目录之外运行远程安装。
