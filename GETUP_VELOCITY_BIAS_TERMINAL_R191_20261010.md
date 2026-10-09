# R191 速度相关惯性偏置完整路径终态

2026年10月10日。独立6完整smoke和正式56已保存闭合结果并自然退出；launcher natural_exit=true、smoke_exit_code=0、formal_exit_code=0、source_hashes_unchanged=true。候选10/24、标准成功、物理无效0，冻结R157基线18/24、标准成功、物理无效0，original_development_gate=false。没有改善、扩大或资格，统一最好仍R15718/24；full_task_completed/hardware_readiness均false。R191没有修正版、搜索或学习器，不重跑本规则或追加幅值/方向/窗口/模型参数预算。

## 实际执行与数量

ROOT=/data/shijinsheng/open_duck，项目ROOT/projects/Open_Duck_Playground，SSH5090x8。启动器现有.venv/bin/python -u -m diagnostics.launch_getup_velocity_bias_r191 --launch；后台实际同launcher无参数。先同python -m diagnostics.test_getup_velocity_bias_r191，14项1.300秒通过；同python -u -m diagnostics.probe_getup_velocity_bias_r191 --smoke独立自然退出后，再14项1.246秒通过，随后同probe无参数正式。OMP/OPENBLAS/MKL各1、CUDA空、JAXcpu、6CPU、MuJoCo3.12.0。

输出ROOT/outputs/getup_velocity_bias_r191_smoke_20261010、getup_velocity_bias_r191_left_20261010及同名.log，启动器getup_velocity_bias_launcher_r191_20261010与ROOT/tmp/getup_velocity_bias_r191_launcher_20261010.log。独立6+正式6启动一致性+25baseline+25candidate=62闭合恢复路径尝试，不是62独立起点。本轮62均2279controls、45.58秒，共141298控制记录，全部物理有效；不外推以前无效早退回合走满。原12秒进入、连续严格30秒和逐500Hz有效不变。seed291仅原初始化/记录，不入kernel；learned_parameters=0/search_trials=0。

## 冻结方法和输入边界

复用不可变R190 kernel SHA256 44d1f4881ac612e66c2a99fa7bceab16ce0fd4657f3d6bb2089077db3f2d07fa。native34只gyro0:3、全部14关节home-relative q6:20、v20:34除.05、因果initial和固定同相位nominal。私有canonical-root位置0/单位quat、根线速度0；gyro经site rotation/angularJacobian重建根角速度。仅kinematics/comPos/jacSite/comVel/RNE(flg_acc=0)，同q及原gravity两次moving与zero velocity差为b；e=(b-bnom)-(bi-bnom0)，十腿raw=+e/kp，原14servo unitgear/fixedgain/affineposition/无actuator内部dynamic，kp13.37Nm/rad。head新增0，head与腿不同分支的限制保留。up3:6为实际framezaxis/worldsite轴，本规则不用，不称bodygravity。

不读actualroot、实际qpos/qvel、case/seed/path/label、teacher、未来实际观测或lookup作新增推理。固定名义不是本回合未来实际数据；原qpos/qvel只原保存和孤立全字段equality。无forward/inverse/积分/contact/distance/passive或实际force查询，不向原physics写force。b不包含gravity、Mqacc、constraints、实际rootlinearvelocity。这是同状态未clip servo代数前馈假设，不是20ms响应、接触/根acceleration、恢复/安全或耗散预测。

冻结R157真实初始native50 selector、R122 snapshot节点、原IMU及右髋反馈；原postIMU/right double目标之后十腿合并，head原样。合计14相对reference±.18，再原joint/5.24rad/s乘.02slew/torque，原50/500Hz/mesh/flags/reward/criteria不变。zero保留原对象、control0原target、标准新增raw/e精确0、home529后新增信号0而原feedback保留。标准b/moving/static可以非零，不称全部偏置零。

保存original_double_pre_slew_target_rad及adjusted double、raw/e/四组b/current moving/static RNE/私有canonical重建速度、pre/post直接计划差。原IMU double调整前request未单独保存，是明确postIMU/right输入边界，不冒称重建全部原IMU。保存信号只作为证据，不是独立全决定审计或动态反事实。

## 完整接口与结果

14项新接口回归覆盖worker冻结snapshot初始化、actualscalar因果签名、529名义/control0零、原对象/head与合计界、原限位顺序/prev历史float32、private禁止动力学、模型完整hash和启动源码捕获，以及门禁拒绝无效或标准非零。没有重复R190离线20项数学验证目录。实际wrapper逐控制断言planned与原step_target applied及sim.prev精确一致。

原constructor40prepare与显式case reset40prepare两批分开保存真实frames/time。异常机制保存完成批次及最新partial/failed causal inputs；本轮无此异常，不新增prepare积分、等待、沉降或中途root reset。旧R188首次80帧异常、b旧launcher phase KeyError和c修正、旧R190错误head分支测试失败及其他R183/R185/R187/R189来源缺口原样保留，不回填或改称新证据。

独立disabled标准/769000/773004全部旧数组/prepare/initHash/peaks逐位R157。enabled标准完整逐位原样，11.08秒进入/34.70秒严格尾段、raw/e=0。enabled769002完整失败有效，raw .003634974940968341rad、post .0036349749409683415rad、合计 .13399598426728154rad。enabled773004成功有效，11.10/34.68秒、raw .006837605094433795rad、post .006837605094433791rad、合计 .13306071774328443rad。不把独立smoke少数成功称统一成绩或安全证明。

正式同6全部新旧数组/initHash/peaks逐位独立smoke；25baseline全部原字段/peaks逐位R157、25candidate/baseline原初始hash配对。候选10成功为769003/769004/769006/769007/773000/773004/773005/773009/773012/773013。救回769004/773004/773005，退化769000/769001/769005/773002/773003/773006/773007/773008/773010/773011/773014。全部失败保留，不排除失败统计。

24扰动保存的同状态post-slew首次差均control1，raw峰值范围 .0023738400696582023至.009215486300679521rad，合计14修正最大 .18rad。773005救回11.12/34.66秒、raw .004205312593274851rad；769000退化raw .008489248602414817rad。只是原保存信号统计，不是瞬时动态传递增益、反事实或唯一根因。不同访问状态随后sensor/原IMU/target分叉不能当同状态即时效果；小raw、未达cap或控制端点不能证明安全，不能浅几何重标原500Hz峰值。

## 来源、归档和发布条件

probe SHA256 8871d666a5ea6a60a472db72f0978404f1c3cc3c2797cbd3d9292b1163bb7fbe；test 7e7f82c1a890e06ea11555332eb4d81bb93c261945f899debc0fba6252d2bcf1；launcher 1c06741ecb5172d225fbc2bee54b1e54120b2d996a5111747a4970577793e58d。本机/server两端核对一致；main、各spawn、两层回归和实际launcher启动显式捕获main及全部已导入Python源码，不是native binary快照。

全部XML include/mesh/fileassets及source/frozen/stand/reference/input前后hash不变，actualcompiledMJB 53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3与旧physics 4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae逐轨迹不变。独立输出53807933bytes、正式179493170bytes、launcher30984895bytes；含新归档和Git仍受2GB预算，保留10GB。终态前只读仅live_server2664890，无R191任务，可用18199343104bytes；这些是当时快照，下一轮必须实时检查，不清理他人或未确认数据。

唯一新archive_getup_velocity_bias_terminal_r191保存两新terminal_snapshot、启动器terminal、全部原失败/labels/peaks/信号/RNG/执行源码/14回归/log/25配对和新docs/scripts，不复制重归档旧dynamic。新固定源码/doc/terminal/archive/publisher/receipt不可复用覆盖。本文写于归档发布之前；只有唯一publisher等待exit0、内部remote核对写成功收据，再原helper独立只读核对同明确新commit/main/local/clean且GitHub完整读取本文之后，才可称发布完成。不得按本地提交或本文准备条件提前宣称发布。不得竞争push、强推、导出key或修改原历史doc。

下一轮发布完成且无R192或更高实际任务，才新唯一编号只读审计62既有轨迹：从actualnative34/因果initial/fixednominal独立重算b/e/raw、gyro/weld/单位/servo，再从原exact postIMU/right double边界重算reference合并/head/joint/slew/base/adjusted/applied/递推prev。已存b/e/raw/planned只比较不用作决定，qpos/qvel仅孤立equality；无env/forward/积分/contact/force或成败物理重标。先独立只读smoke包含标准、非零救回、退化、失败，再自然exit和正式全62核验，source/evidence/model前后hash不变。不给本速度偏置参数追加预算，不重复R191动态或R190验证。

当前18/24不跳扩大316/未见320；320仍未读取执行。原22/24→316>=36/40→320两个20组各18/20和后续更大独立扰动、sensor/executiondelay及右/俯/仰门槛不变。保护live_server2664890，无硬件连接/使能/部署，原walker不改。保持ACTIVE30分钟健康静默续跑；仅全部仿真验收通过且存证/列硬件未过项后才结束。
