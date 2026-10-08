# R170 因果已执行目标跟踪记忆：回归、独立完整冒烟和正式闭合启动

2026-10-08。前置终态为 `f741d1b8fa07448899988bd9c63f75f59e9033d0` 的 R168/R169；本记录不把新启动称为终态或验收改善。当前统一已证最高仍 R157 18/24，原 22/24 门槛未通过，未扩大、未读取执行 3200000 至 3200039 未见资格。

## 实际任务与状态

本轮开始只读核对 GitHub main 和服务器干净发布库均为上述提交，未发现 R170 或更高实际任务，无 getup 训练/审计/探针进程；保留 `open_duck_agent.live_server`。创建新源码和唯一目录，不修改旧执行源、失败、模型、归档或原行走控制器。

项目 `/data/shijinsheng/open_duck/projects/Open_Duck_Playground`，现有 `.venv`。2026-10-08 08:11 UTC 实际完整正式命令：

```text
/data/shijinsheng/open_duck/projects/Open_Duck_Playground/.venv/bin/python -u -m diagnostics.train_getup_tracking_memory_r170
```

无额外参数。安全启动器 `diagnostics.launch_getup_tracking_memory_r170` 设置 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu`，重新十四项回归、确认无同类动态任务且正式目录唯一后启动。启动 PID 3468062 只供定位，必须实时核对完整命令，不按旧 PID 操作。本记录时正式六条启动回放已闭合，`startup_closed.json` 已保存；`progress.json`、`training_closed.json`、`results.json` 尚不存在。主进程正常，正在第一代候选评估，不能当训练结束。

独立冒烟命令：

```text
.venv/bin/python -u -m diagnostics.train_getup_tracking_memory_r170 --smoke --output /data/shijinsheng/open_duck/outputs/getup_tracking_memory_r170_smoke_20261008
```

独立六条全部完整 2279 控制步，保存结果后 Python 自然延迟退出，等待进程自然退出后才正式启动；无信号、重启或强制恢复。正式目录 `outputs/getup_tracking_memory_r170_left_20261008`，同名 `.log`；独立目录 `outputs/getup_tracking_memory_r170_smoke_20261008`，同名 `.log`。

## 证据、区别与有限假设

R165 至 R169 的瞬时速度修正和同状态专家输出混合在不同案例出现救回及更多退化，微小新增也可能伴随原反馈与姿态随后分叉。它们没有证明唯一原因、所有反馈无用，或成功专家在另一访问状态可恢复。R170 不重复方向统计或阈值，不改旧三速度范围/符号或 48 混合矩阵预算。

新假设是检验执行目标跟踪历史的有界记忆是否比即时误差修正更适合这些耦合：同一当前读数和初始上下文，因过去的跟踪过程不同可以有不同控制器状态。它不是旧单侧六位置/速度瞬时系数、双侧十二共同差动系数、静态成功集合拟合、BC 聚合、专家混合或相位调整。仅检验新结构，不声称已解决失败原因或改善。

R157 实际初始 native50 选择器、R122snapshot 程序/节点、原当前右髋反馈及 IMU 反馈完全冻结。推理新增只读当前和因果初始实际 14 关节位置、实际前一步已执行目标及内部过去记忆；使用同相位名义真实传感器作相减。没有根位置/线速度、case、seed、目录、标签、教师选择、未来子步、精确上下文或近邻查表输入。case 只用于原 episode 初始化与日志，原历史成功标签不用于新反馈训练监督。

native55 的 `6:20` 是关节位置减 home，`34:48` 是 `sim.prev-home`，两者均 rad。每次原 `step_target` 更新之前因果采样，并断言目标字段等于当时实际 `sim.prev-home` 的 float32 值。**不使用未 slew 的 `history[0]` 或当步尚未执行的新目标作为跟踪读数**。新状态不是物理状态，不新增准备/控制积分。

## 具体数值函数

每通道跟踪误差 `r = actual_previous_applied_target - actual_joint_position`；先减同相位名义对应值，再减回合开始因果实际对应误差，得到 `delta`（rad）。固定尺度 `.05rad`，`drive=tanh(delta/.05)`。记忆从零开始：

```text
z_leaked = exp(-0.02 / 1.0) * z_previous
z_proposal = clip(z_leaked + 0.02 * drive, -1, 1)
leg_extra = +0.18 * tanh(nonnegative_side_coefficient * z)
```

只训练左右五腿各一个非负幅值 `0..0.05`。固定一秒泄漏常数和记忆界 ±1，不搜索泄漏、记忆界或时间窗；头颈四通道 drive、memory、新动作始终零。模型解析 actuator 名称，左/右/头颈数量 5/5/4，不假定左右执行器是连续索引；原右髋 native 索引 9/10/11 单独核对。

条件防累积饱和只作当下目标规划：在同一当前状态分别通过原 joint/slew 限制计算原目标和新增目标；若未能执行的新增目标分量与当前 drive 同向（乘积大于固定 `1e-13` 舍入界），撤回该通道本步积分增量，保留已泄漏旧记忆；反向 drive 允许记忆释放。然后重算新增并经原合并/限制执行。它不读取未来动力学，也不证明机械能耗散、接触安全、不会碰撞或成功恢复。

控制 0 新增与记忆精确零，第一目标原样；标准 actual scalar 全程误差变化/drive/记忆/新增精确零。零系数输出零，原目标合并对象直接保留，不重算原反馈。529 步起身后 home 原反馈，控制器内部记忆归零但根状态不重置。原新合计全部 14 joint 相对参考各 ±.18rad，原 joint、torque、slew、home、mesh、flags、50Hz 控制/500Hz 物理、reward/acceptance 不变；完整回合无根重置/传送、额外准备积分或等待。

## 回归和实测完整冒烟

十四项回归首次及安全启动器重新执行均通过：因果函数签名；标准 scalar 精确零；非零初始误差首步零；native 位置/已执行目标 rad 单位和符号；未读 IMU/速度/接触等通道无影响；同当前读数不同历史记忆及固定泄漏；长递归有界/头颈零；左右组独立；zero 合并对象；cap 阻挡撤回；允许释放；当前 slew 阻挡；随机总目标/joint 边界；非法系数/NaN/记忆界拒绝。本轮没有回归失败或执行源码修正版。

独立 zero 的标准、769000、773004 全部原 observations/actions/time/qpos/qvel/applied/strict/local feedback/prepare 数组、初始 hash 和原 peaks 逐位复现 R157。三条分别标准成功、769000 成功、773004 失败，全部物理有效。零幅值时扰动内部记忆可非零，但动作不变，不把它误称扰动记忆全零。

固定预定 `[.003,.003]` 非零冒烟：

| 起点 | 原完整成败/物理 | 进入/严格尾站稳 | 最大记忆 | 最大新增 rad | 撤回通道步数 |
|---|---|---|---|---|---|
| 标准 | 成功/有效，原全轨迹逐位 | 11.08s/34.70s | 0 | 0 | 0 |
| 769002 | 失败/有效，完整 2279 步 | 未进入/0s | .50244946 | .00027132250 | 499 |
| 773004 | 成功/有效，完整 2279 步 | 11.12s/34.66s | .15282581 | .00008252593 | 248 |

两扰动真实新增非零、头颈新增零、第一目标原样。773004 仅已用开发冒烟救回，不是统一候选成功数、泛化或资格，769002 失败完整保留，不调 PROBE 或用单例外推。

正式又同六条完整启动回放，所有保存数组（含新记忆/drive/撤回/提案/新增字段）、初始 hash 和物理 peaks 与独立冒烟逐位一致，已写入 `startup_closed.json`。这十二动态回放只有标准及已用扰动，不能称十二独立起点。

## 有界训练和后续门槛

seed270，三代、每代八提案、六 CPU；两非负幅值 0..05，CEM mean0/std.003、elite2、.6/.4 更新、std 限 `.0003..01`。每代保留 zero/最佳，缓存重复参数，全部实际候选、失败、参数、分布/history/RNG 与闭合 checkpoint 保存。每套实际程序在标准+原24真实完整倒地按原2279步45.58秒、12秒进入/严格连续30秒/逐500Hz物理有效评估排序；物理无效可早退，不能称每个尝试都走满。不用629步短尾代替验收。

结束自动候选与 zero 新增反馈基线各25完整开发，初始 hash 配对，baseline 必须逐位 R15718/24 原字段和原 peaks。只有标准保留、原>=22/24、优于本轮18基线且全部有效，才另新目录冻结扩大316>=36/40、优于fixedzero/R102且全有效，然后冻结未见资格。父进程不自动扩大/资格。316/318已开发，320未读取执行；资格两个不重叠20组各18/20、12秒/连续严格30秒/500Hz全有效，以及后续更大扰动、延迟和右侧/俯卧/仰卧验收仍全部需要。

未改善先完整实际记忆、tracking error/drive、撤回、同状态直接新增与随后原反馈/姿态响应、救回退化及原物理无效审计，不补同PROBE、追加同两记忆幅值预算或回到旧禁止结构。不能以小动作、理论界、单例冒烟或学习损失代替完整验收。保持ACTIVE，不连接硬件、不部署，不称真机可用。

## 保存边界

新 `archive_getup_tracking_memory_startup_r170` 只追加独立 `terminal_snapshot` 六闭合轨迹及正式 `startup_closed_01` 六闭合轨迹/冻结源模型物理哈希/十四回归日志。正式快照 `terminal_result_saved=false, training_generations_saved=0`，不拿开放训练候选或后来 progress 当终态；独立 smoke 终态仅表示冒烟闭合，`full_task_completed=false, hardware_readiness=false`。实际新闭合代数和最终结果必须另唯一目录/新脚本保存，旧startup/g1/terminal执行源、失败和bundle不可覆盖。发布前后核对干净库与最新main，bundle完整大小/hash及依赖校验后正常快进，不强推、不上传凭据。服务器只读剩余约33GB，不清理他人或未确认数据。
