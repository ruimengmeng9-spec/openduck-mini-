# R197：独立双价值网络与经验回放起身训练

## 目标与当前状态

用全新初始化的离策略双价值网络学习起身阶段的十腿总目标反馈，验证它能否在共同复测中超过冻结的 R157 18/24。计划冻结时尚未启动动态回合；回归、独立冒烟、正式训练的结果以新目录的实际记录和自然退出回执为准。不预称改善、安全或完成验收。

R196 两轮确定性结果为 7/24 与 8/24，最终 18/24 是 R157 回退，不是新 PPO 成绩。R195 辨识门禁失败、MPC 未动态启用。两者均已闭合、归档和发布，不重复动态、核验、归档或上传，不扩大任何旧方法预算。

## 结构区别与输入边界

这是受 [TD3 原论文](https://proceedings.mlr.press/v80/fujimoto18a.html)启发的有限变体：两个独立动作价值网络、实际新转移的经验回放、一步 Bellman 目标、较慢目标网络和延迟确定性策略梯度。它不是 R196 的完整回合 MC/PPO 概率比更新，也不是 R181 整回合 Gaussian score；不复用其权重、Adam、训练轨迹或批次。双价值网络只是回报估计，不是已验证的物理效应模型或安全证明。

在线输入仅为当前实际 native50、同回合因果初始 native50，各逐项 tanh，加固定控制进度 min(k/528,1)，共101维。没有 case、seed、路径、标签、root、qpos/qvel、teacher、未来实际观测或 lookup；原 framezaxis 不称 bodygravity。训练 seed297 仅用于新初始化和离线随机数，不是在线标签。

actor 为101→32→16→10，隐藏层及输出 tanh；输出层权重/偏置初始零，其他权重新随机，全部可学习。两个独立 Q 为111→32→16→1，输入为101特征加10维请求动作；隐藏 tanh、输出线性，各自权重新随机，偏置零。不存在旧模型 warm start。

actor 输出归一化请求 a；采样 a=clip(actor(x)+1e-4*创新,-1,1)。控制1..528的十腿目标为 reference+.18*a，随后原 joint/slew/torque 限制。不是对 a 再 tanh，不加旧腿反馈，也不把执行后动作称高斯。control0 原目标，四 head 原样，home529+ 原反馈；R157/R122/IMU/right 只作为同状态旁路、首步、头部与 home 原项。

## 回放、原奖励与固定更新

只用本轮新训练回合构建回放，第一轮用第一轮，第二轮用第一加第二轮。每个失败和原 invalid 回合均保留；不筛掉失败、不用成功并集，不将 baseline、旧轨迹或 teacher 数据加入学习。原奖励和终端奖励不变。

只收集恢复阶段 k1..528 的转移。下一真实观测在到达后才供离线目标估计；在线控制不读下一状态。gamma=.99**.2：

- 下一步仍在恢复段且本回合未结束：y=r+gamma*min(Q1target,Q2target)。
- k528 后进入 home：y=r+gamma*本次记录的完整原 home 折扣回报，不学习 home 动作，不丢掉其奖励。
- invalid/终端提前结束：y=r，不跨 reset bootstrap；无效路径标签和500Hz原峰值保持原样。

每轮固定64次双 Q 更新，每批512条有放回抽样；每2次 Q 更新执行1次 actor 更新。总计128次 Q、64次 actor 更新，不追加迭代。Q 学习率1e-4，actor1e-6；各优化器独立 Adam、globalnorm1。actor 目标为 -mean Q1(x,actor(x))，更新时 Q 参数固定。Bellman目标 stopgradient。

目标动作噪声1e-4，噪声限制±2e-4，归一化动作限制±1；目标网络初始等于在线网络，每次延迟 actor 更新后 tau=.005 Polyak 更新。参数、噪声、轮数、网络或窗口不能在失败后追加搜索。

保存初始和两轮完整在线/目标权重、两套 Adam 状态/count、实验 RNG、每轮完整新回放、采样索引和目标噪声、所有更新损失/梯度范数、最后完整梯度和参数变化。能离线重算学习器，但没有可直接恢复在线环境的完整状态，不冒称跨实现 bitwise、学习有效或安全。

## 固定动态与存储预算

最多162次闭合路径尝试，不是162个独立起点：

| 用途 | 次数 |
|---|---:|
| 独立完整冒烟 | 6 |
| 正式同6启动一致性 | 6 |
| 原 baseline（标准+24） | 25 |
| 两轮采样训练 | 50 |
| 两轮共同确定性复测 | 50 |
| 最终 candidate 或 R157 回退 | 25 |
| 合计 | 162 |

正式最多156次。每条原2279controls/45.58s，invalid允许原早退并保存，不声称全部走满。最多6训练CPU，OMP/OPENBLAS/MKL各1、CUDA空、JAXcpu。原 constructor40和case reset40两批实际准备帧分别记录；不增加积分、等待、沉降或中途 root 重置。

live+新归档+Git增量总预算2GiB；启动free至少12GiB，全程保留10GiB。新自动存储检查包含 preflight、launcher、smoke、formal，以3倍 live保守预留归档与Git；归档前还须独立统计实际三项。碰到门禁先保存失败，不能删除他人/未确认数据或用“剩余试次”继续失败方法。历史free只是快照，启动前重新读取。

## 执行门禁与证据

先离线回归，涵盖因果特征、初始零 actor、独立 Q、动作界、十腿替换非叠加/head不变、原joint/slew、完整home奖励、终端截止、minQ目标、目标平滑/stopgradient、actor数值梯度、固定更新/延迟、checkpoint状态、smoke null标准处理、预算和禁输入。测试不构造动态环境，保存四显式源码及已导入Python副本和映射 manifest，不补写 R196 缺失 manifest。

独立6：baseline标准/769000/773004；新初始确定性标准；新初始采样769002/773004。原 baseline 全旧字段/initHash/peaks逐位R157；新标准须成功且全部6物理有效、完整2279controls。扰动组可起身失败，不能放宽 physical 门禁。新标准无需逐位R157，这是既有“仅替换起身旧反馈”授权；初始 actor0 只相对 reference0。

必须等独立进程自然exit0，再回归及 idle检查，通过门禁才由唯一launcher运行正式任务。正式同6全部新旧数组/result/RNG须精确一致。25baseline须18/24、标准成功、全valid；候选资格须标准/allvalid、保留baseline全部成功且严格超过18，按成功数/min原return/sum原return选；无合格选择R157并保存最终25，不把回退成绩叫TD3成绩。

实际 applied/prev 与从原边界重算的 planned 逐步精确。postIMU/right double 是明确边界；未保存的 before-request/完整旧 IMU double 不冒称独立重建。qpos/qvel只原记录及隔离 equality，不进控制或价值学习。保存全部原失败/labels/peaks/初始hash，不用小动作/cap/控制端点/浅geometry重标500Hzinvalid，也不把访问状态分叉当即时反事实或唯一根因。

完整XML/assets、冻结输入、执行源、MJB与原physics前后不变。所有新源/失败/终态/收据执行后不可覆盖；若实现缺陷，需要另编号保留首失败，不改物理或验收。保护 live_server2664890，不信号/重启，真机不连接、使能或部署。

## 不变的验收与后续

统一 best仍R15718/24；原24>=22且>18/allvalid才能进入新目录冻结316>=36/40，再冻结未见320两组20各>=18，原<=12s进入/连续strict30s/逐500Hz有效及更大扰动、传感/执行延迟、right/prone/supine全部门槛保持。320仍未读或执行。本轮不自动扩大资格；全部仿真门槛未过时 fulltask/hardware/qualification=false。

服务器ROOT=/data/shijinsheng/open_duck，项目projects/Open_Duck_Playground，SSH5090x8。唯一新launcher为 diagnostics.launch_getup_total_target_td3_r197 --launch，后台同无参；子 train_getup_total_target_td3_r197 --smoke 自然退出后同无参formal。新目录为 getup_total_target_td3_r197_preflight_20261010、getup_total_target_td3_launcher_r197_20261010、getup_total_target_td3_r197_smoke_20261010、getup_total_target_td3_r197_left_20261010。R196发布HEAD0c56db66bc246ab8649b9f130475400f3390ec34且clean作为启动基线，实际状态另外记录，不改其归档前历史条件。

自动化保留当前实际设置，不用旧ACTIVE恢复将来的人工暂停。只在实质进展、异常、完成或需用户处理时通知。

