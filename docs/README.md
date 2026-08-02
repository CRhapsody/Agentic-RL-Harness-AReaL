# 课程索引

这组课程从可运行代码出发，逐步解释 Agentic RL、AReaL、Hermes 与可学习 Harness 之间的数据关系。

## 已完成

1. [第二课：一条 Agent 轨迹怎样变成 AReaL PPO 训练数据](lesson-02-agent-trajectory-to-areal-ppo.md)
2. [第三课：一份 episode reward 为什么必须形成两条更新链](lesson-03-one-reward-two-update-chains.md)
3. [第四课：一次真实 AReaL rollout 怎样与 Harness 决策对齐](lesson-04-real-areal-harness-bridge.md)
4. [第五课：从可信 AReaL rollout 到真正同时更新 policy 与 Harness](lesson-05-from-trusted-rollout-to-joint-update.md)

第二课使用本地 `AReaL v2.0.0` 的固定提交和本项目 calculator smoke 作为唯一代码依据。课程中的 token ID 与 log-prob 数值均明确标为教学例子，不代表当前 mock 运行的真实产物。

第三课对照 G1 synthetic CPU 控制面实现，讲解 policy token 与 Harness decision 的独立概率、mask、credit、版本、发布和恢复边界。它不把 toy updater 表述为真实 AReaL 联合训练。

第四课对照项目提交 `41e00d9a2215d03c1108d9728d0a4a8c20752a7a`，解释真实 AReaL 六字段、Harness prompt、request sidecar、同 controller log-prob 复算、score token-ID 精确绑定和 `JointVersion` 怎样连接。v5 GPU 复验的概率门仅 `1/4` 轨迹通过，`audit.json` 未生成；它不包含 optimizer update，也不宣称已经完成 policy 与 Harness 联合学习。

第五课把可信 rollout 之后仍缺少的训练事务展开为可检查对象：冻结联合 batch、两类 old log-prob 与行为版本、两路 credit、两个 optimizer、候选 checkpoint 和原子联合发布。它穿插对照本项目的真实 bridge、G1 synthetic 控制面、AReaL `PPOTrainer` 与 Hermes session 路由，并明确说明正在执行的 C0/C1 只校准数据面，不是 optimizer 或联合发布证据。
