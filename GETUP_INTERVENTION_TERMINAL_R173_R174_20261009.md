# R173 动作干预模型训练与 R174 成对效应审计终态

2026年10月9日。R173七套固定程序的175次完整路径尝试和一次离线模型拟合已结束并自然退出；R174独立只读冒烟及正式审计均已结束。它们没有产生新的起身控制策略。统一已证最好仍为R157的18/24、标准成功、物理无效0，原开发门槛未通过，未扩大开发或执行未见资格，硬件可用性未通过。

旧启动记录`GETUP_INTERVENTION_DYNAMICS_R173_20261009.md`和`startup_closed_01`保留当时状态，不覆盖或冒称终态。本次使用新的`terminal_snapshot`保存全部闭合尝试、学习器、失败和审计；原`progress.json`仍保留历史terminal=false，最终状态以`results.json`、`training_closed.json`为准。

## 完整命令和固定方法

项目`/data/shijinsheng/open_duck/projects/Open_Duck_Playground`，使用现有`.venv`。OMP、OPENBLAS、MKL各1，CUDA_VISIBLE_DEVICES空，JAX_PLATFORMS=cpu。

```text
.venv/bin/python -u -m diagnostics.launch_getup_intervention_dynamics_r173 --smoke
.venv/bin/python -u -m diagnostics.train_getup_intervention_dynamics_r173 --smoke
.venv/bin/python -u -m diagnostics.launch_getup_intervention_dynamics_r173
.venv/bin/python -u -m diagnostics.train_getup_intervention_dynamics_r173
.venv/bin/python -u -m diagnostics.audit_getup_intervention_dynamics_r174 --smoke
.venv/bin/python -u -m diagnostics.audit_getup_intervention_dynamics_r174
```

R173冻结R157实际初始native50选择器、R122snapshot程序节点和原IMU及右髋反馈。seed273提前生成统一529步三路符号序列：zero、正负yaw、正负roll、正负pitch共七套。请求最大幅度固定0.0005rad，乘当前native34相对同相位名义传感器固定尺度偏差的tanh，使名义精确零；控制0和home新增零。没有幅度、方向、时间窗、特征或正则网格搜索，干预程序不作为起身候选优化或提升。

每次从原完整倒地独立初始化，不中途复制状态、根重置、传送或额外prepare积分。实际控制前保存规划目标，逐控制断言其与原step_target返回目标、slew后sim.prev逐位一致；native历史目标在原入口更新之前采样。保存同状态未干预规划目标、实际执行差、请求新增和完整下一读数。总14joint相对参考仍各±0.18rad，随后原joint范围和5.24rad/s slew不变，home原反馈、50Hz控制和500Hz物理不变。原mesh、flags、力矩、奖励、验收和walker未改；每回合物理指纹前后相同。

输入只含当前/过去真实native传感器、同相位名义数据和冻结的统一控制序列。case、seed、目录、标签、root真值、未来子步、最近邻或精确上下文不作推理输入。原完整轨迹qpos/qvel仅审计保存；未来下一帧只作离线监督目标。

## 闭合回合与原验收

每个正式程序标准加原24开发起点，共175次原2279步、45.58秒路径尝试，按12秒进入、连续严格站稳30秒和逐500Hz物理有效标记；无效可提前退出，不能称175次均走满。另有6次正式启动一致性回放，合计181次正式动态尝试；旧独立6次冒烟合计187次，不是187个独立起点。

| 固定干预程序 | 原24成功数 | 物理无效数 |
| --- | ---: | ---: |
| zero | 18 | 0 |
| 正yaw | 6 | 0 |
| 负yaw | 8 | 3 |
| 正roll | 8 | 1 |
| 负roll | 9 | 0 |
| 正pitch | 5 | 4 |
| 负pitch | 8 | 2 |

七程序均保留标准；zero全部25轨迹的原保存字段、prepare、初始化hash及原物理峰值逐位复现R157。12类干预回归和原R172模型10项数学检查、独立六次完整冒烟、正式六次新旧数组逐位一致性通过。各程序25回合检查点、全部失败、固定符号序列、参数、生成后RNG与执行源码闭合。七程序的首干预前共同前缀一致，保存25起点乘3轴的75项配对。只有此时声明同状态，后续分叉不当同状态即时动作效果。

## 一次离线学习器与选择偏差

全部数据闭合后，沿用固定342维当前/前帧native50、实际当前规划目标14和相位特征，float64 ridge1正常方程一次求解及动作输入消融。新数据来自明确指定的动态干预，不是旧被动MSE预算扩展。固定留出769000、769001、773000、773001和pitch正负轴；它们都是已开发数据，不是资格。

175回合中10个物理无效回合完整保留，但整回合排除拟合。这造成选择偏差，模型不能外推这些无效访问状态。有效输入165轨迹，训练101轨迹、230078个转移，全部165有效轨迹保存预测和阶段误差。`learner_closed_01`保存模型、动作消融权重、正常矩阵/右端、数据manifest、固定solver/seed/RNG和评估。

模型SHA256为`a02e23535a56e0d69accb9a862cc82b76b27d349098564574b9afc2dce64bf07`。这是传感器一步预测器，不是已验收预测控制器。

## R174 首次干预效应核验

独立只读冒烟为标准、769000、773007各3轴共9配对，自然退出后正式25起点各3轴共75配对；对应冒烟行与正式逐位相同。正式72个扰动配对在首干预控制1的实际规划动作差均非零，当前和过去状态、首目标与未干预计划逐位相同，正负请求反对称。比较下一控制帧native34差与新模型、旧R172b模型和动作消融的预测差；这是20ms末传感器效应，不是首次500Hz越限子步或原碰撞峰值。

下表为72个扰动配对的逐配对RMSE算术均值，不是池化所有样本RMSE。动作消融预测差为零，其误差因而等于真实成对效应相对零的误差。

| 传感器与单位 | 新模型 | 旧R172b模型 | 动作消融零效应 |
| --- | ---: | ---: | ---: |
| gyro rad/s | 0.00106079 | 0.000735995 | 0.0000451881 |
| upvector | 0.0000296068 | 0.0000223769 | 0.000000235726 |
| 关节位置 rad | 0.0000109392 | 0.0000109552 | 0.00000237177 |
| 关节速度 rad/s | 0.000852868 | 0.000798881 | 0.000208747 |

这次固定小干预下，新旧模型都未可靠预测首次实际动作效应，不能仅凭完整轨迹平均误差较保持基线低就支持在线预测控制。它不证明所有动力学模型或反馈无用，也不授权扩大同一幅度/特征/ridge预算来追分。

R174同时保存case_held、axis_held、both_held的早段/恢复/home误差；各指标平均是逐轨迹均值，峰值取留出轨迹中最大绝对误差，不用峰值均值掩盖极端值。both_held八条非标准有效轨迹的恢复段新模型gyro RMSE约0.19820rad/s，动作消融约0.19841rad/s；新模型最大gyro误差2.05733rad/s、最大关节速度误差2.62085rad/s。不能把这类误差当30秒起身成绩。

只读审计不构造env/MjData、不forward、不积分、不推断力、不重标成败或物理标签。scalar只解码白名单观测及规划/请求字段；原压缩轨迹文件SHA256会读取完整文件字节，但不解码root数组，不能把此范围写成完全不读取含root的文件字节。源码/模型/源轨迹hash前后不变。正式及smoke的pairs、75和9份信号数组、误差、原标签与源hash完整保留。

## 追加归档和发布

R173输出`/data/shijinsheng/open_duck/outputs/getup_intervention_dynamics_r173_20261009`；R174正式及smoke为`getup_intervention_dynamics_audit_r174[_smoke]_20261009`。本次分别追加新`terminal_snapshot`，不覆盖R173已发布smoke或startup快照。归档复制前后逐文件hash相同，包含全部闭合程序、失败、学习器/RNG、原回归日志、过程日志和执行源。

2026年10月9日连接恢复后，先正常快进上传已校验的R172及R173startup，GitHub push返回main从`0e24009b357eda680fb9db7042587d2de8a5ef17`更新至`292ed31106f81bc0b8fc61a0bab236a5fd58b2c1`。新的终态归档以干净服务器发布库该提交为base，后续实际commit、唯一bundle大小/SHA256、完整SCP、依赖verify、fetch、push和独立远端核对以工具输出为准；不把传输中或仅本地commit当发布成功。

当前开发>=22/24且优于18基线、标准保留及全部有效门槛未过。仍须之后冻结扩大316>=36/40且优于fixedzero/R102全部有效，才可冻结未见320四十例、两组二十各>=18/20及原12秒/30秒/逐500Hz条件。316/318已开发，320未读取执行；更大扰动、传感器/执行延迟、右侧/俯卧/仰卧和硬件验收未通过。本次只归档发布，不新启动训练、连接真机、操作旧PID或停止live_server。服务器本次只读/data约29GB，不删除未确认数据或上传凭据。
