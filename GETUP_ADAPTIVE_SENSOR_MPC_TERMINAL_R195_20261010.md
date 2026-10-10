# 在线自适应传感模型起身实验终态

2026年10月10日。用户授权的新方向已完成R195独立8次完整动态辨识冒烟，并自然退出。22项新回归全部通过，1.347秒。四组动作效应预测均未通过预先冻结的成对门禁，正式辨识、MPC独立冒烟和MPC候选复测均未启动。统一最好仍R15718/24，无改善、扩大或资格。全部仿真验收未通过，真机没有连接、使能或部署。

## 实际动态和冻结方法

项目为/data/shijinsheng/open_duck/projects/Open_Duck_Playground，现有.venv，SSH5090x8。OMP/OPENBLAS/MKL各1、CUDA空、JAXcpu，最多6CPU。实际启动diagnostics.launch_getup_adaptive_sensor_mpc_r195 --launch，后台同模块无参数；子进程diagnostics.train_getup_adaptive_sensor_mpc_r195 --smoke。子进程自然exit0，launcher natural_exit=true、smoke_exit_code=0、formal_exit_code=null、formal_not_run=true、source_hashes_unchanged=true。formal目录不存在。结果先保存后自然退出的等待期间没有重启、发信号或按旧PID操作。

输出getup_adaptive_sensor_mpc_r195_smoke_20261010及同名log，启动器getup_adaptive_sensor_mpc_launcher_r195_20261010，日志tmp/getup_adaptive_sensor_mpc_r195_launcher_20261010.log。八次为zero标准/769000/773004、启用校准标准、正负校准各769002/773004。全部2279控制、45.58秒、原物理有效，共18232控制记录；不是八个独立起点。四个非零校准路径原起身均失败，不能当24案例共同成绩。zero原标准和769000成功、773004失败有效；启用标准成功且新增精确零。全部失败、原完整result、peaks、初始化hash和真实prepare保留。

新模型不是R172/R173的342维静态ridge：每回合独立零权重和100I，78输入为native34因果偏差、上次34偏差增量和已执行十腿动作差；遗忘0.995，只在下一真实读数到达后更新。五条启用路径各528次更新，共2640次，包括标准零残差更新。没有旧模型权重、成功标签、teacher动作或root监督拟合；无在线case/seed/path/label/root/qposqvel/未来/lookup。seed295只生成共同创新和保留探针。

校准创新标准差1e-5rad乘tanh因果偏差RMS，各腿clip1e-4rad。正负程序仅控制64加减一次1e-5rad共同单位探针，控制64当前观测及此前全部真实历史逐位相同；控制65下一读数差为此次同状态20ms动作效应。后来访问分叉不是即时反事实或唯一根因。framezaxis不冒称bodygravity，原native速度单位不改。

## 成对动作效应门禁

两个扰动配对都走满且有效，真实执行动作差非零，没有数值fallback。独立RLS按实际传感器和真实执行差重算到控制64，权重及协方差与保存值精确相同；保存权重仅compare，不作为独立预测的决定。预测不读取控制65未来读数；未来真实读数只用于已发生效应比较。

预先门禁为四组pooled平方误差分别小于零效应误差的0.8。下表为固定尺度native34偏差的平方误差，非物理安全指标或恢复成绩。

| 通道 | 零效应预测误差 | 模型动作预测误差 | 门禁 |
| --- | ---: | ---: | --- |
| gyro | 6.963337314166059e-12 | 8.532471571340555e-5 | 未过 |
| framezaxis | 1.1842378929335004e-13 | 2.1554510277148385e-6 | 未过 |
| q | 2.3600169437746186e-12 | 3.384876696060934e-6 | 未过 |
| native velocity | 4.2599121155165573e-11 | 4.410801265572173e-5 | 未过 |

模型显著差于零效应，不能据此启用预测控制。这否定本轮有限规则的启用条件，不证明所有在线辨识或MPC无用，也不授权追加幅度、方向、模式、窗口、参数或轮数。未使用的预算不是绕过门禁的许可。

## 接口源码与保留缺口

保留R157实际初始native50selector/R122snapshotnodes/原IMU和rightfeedback。十腿新增、head原样；14合计reference±0.18，再原joint/5.24乘0.02slew/torque；zero原对象、control0、标准新零、home529新增零但原feedback保留。constructor40与reset40两批真实prepare/time分开，无额外积分、waitsettle或rootreset。所有planned与实际applied/prev在原接口逐步断言精确。

postIMU/right exactdouble是明确边界，原before-request及IMUdouble未保存，不冒称重建全部旧IMU。qpos/qvel只原记录及孤立旧字段equality，不入辨识或控制。不用小动作、cap、控制端点或浅几何重标500Hz peaks。模型和传感器访问分叉不是即时反事实、恢复保证或唯一根因。

四固定源码本机/server和执行副本一致：kernel549eddb9f3dada37aae7b1223d9231aad795b06179b40bc2201343540eb79685；main9cdbcf17e407bc4955aa67da98ddb69e9c8dea3262e146202e252693d96dca03；test445a79cc27e428deb56b6fd66e54e776ec590466a5ad32235e9c47b2ed8e2842；launcher7dd8b8db5c036531c91d63f7427b1709e5894b399a8e4891da758daa1caa1eae。main、spawn、test、launcher显式及全部导入Python保存，不是native binary快照。完整XMLinclude/meshassets/frozen/source前后same，MJB53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3和physics4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae不变。

launcher另有过严实现缺口：两个仅扰动的group没有标准，nominal_success为null，被通用标准检查当false；22回归没有覆盖此值。原源码保留未修订或重跑。独立成对门禁本身同时四项失败，因此没有依据开启正式训练；不把错误标准检查冒称四项独立预测结果的唯一来源。

首次归档前核验R195错误调用init_worker读取冻结输入，同时向已闭合smoke追加了核验进程1227578的源码副本和manifest。它不是动态worker启动记录，也不是回填动态source；没有新动态或覆写已有轨迹/result字节。原核验源码、首核验结果、追加副本和verification_source_capture_gap_01.json保留。新archive_getup_adaptive_sensor_mpc_terminal_r195b只改为直接只读加载和缓存非root比较数组，模型/控制/物理/验收不改。新的核验结果以实际--verify自然exit输出为准，不靠旧核验收据替代。

## 追加保存和后续限制

固定总cap116，实际仅8；辨识独立8加正式80、后续MPC独立3加正式25的阶段未超预算，后两段都未运行。原2GiB live加archive加Git预算、至少10GiB保留不变。仅追加本轮闭合输出、源码/失败/完整更新终态/成对预测/RNG/回归/新记录和首核验来源，旧R193/R194动态不复制，不覆写旧证据。发布须新的干净commit、唯一wrapper真实exit0收据、独立readonly远端核对及GitHub明确commit新doc完整读取后才称成功。

原标准保留、24至少22且优于18/allvalid、316至少36/40且优于fixedzero/R102/allvalid、未见320两组20各18及12秒进入/连续strict30秒/逐500Hz、后续更大扰动/延迟/right/prone/supine门槛保持。320未读执行，fulltask/hardware/qualification均false。保护live_server，不改walker/physics/reward/验收，不自动部署。

本轮检查时定期自动化实际PAUSED，保持该设置未自行恢复；此前ACTIVE历史描述不能替代实时设置。本次用户明确授权的手动实验已执行。失败规则不自动追加预算，下步必须真正不同的有限设计。
