# 课程索引

这组课程从可运行代码出发，逐步解释 Agentic RL、AReaL、Hermes 与可学习 Harness 之间的数据关系。

## 已完成

1. [第二课：一条 Agent 轨迹怎样变成 AReaL PPO 训练数据](lesson-02-agent-trajectory-to-areal-ppo.md)
2. [第三课：一份 episode reward 为什么必须形成两条更新链](lesson-03-one-reward-two-update-chains.md)

第二课使用本地 `AReaL v2.0.0` 的固定提交和本项目 calculator smoke 作为唯一代码依据。课程中的 token ID 与 log-prob 数值均明确标为教学例子，不代表当前 mock 运行的真实产物。

第三课对照 G1 synthetic CPU 控制面实现，讲解 policy token 与 Harness decision 的独立概率、mask、credit、版本、发布和恢复边界。它不把 toy updater 表述为真实 AReaL 联合训练。
