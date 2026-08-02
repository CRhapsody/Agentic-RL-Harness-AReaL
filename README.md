# JPH-RL

JPH-RL 是 `PROJECT_PLAN.md` 的最小可执行实现。当前阶段只覆盖 B0：在一个确定性 calculator 工具任务上验证 Harness、模型调用、工具执行、评价器、联合版本和轨迹记录能够端到端工作。

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
