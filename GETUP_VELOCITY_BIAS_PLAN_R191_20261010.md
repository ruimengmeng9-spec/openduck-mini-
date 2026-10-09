# R191 速度相关惯性偏置完整仿真计划

2026年10月10日，R190b只有离线20项通过，没有完整动态验证。R191复用不可变kernel 44d1f4881ac612e66c2a99fa7bceab16ce0fd4657f3d6bb2089077db3f2d07fa，接入原恢复路径检验同状态速度偏置补偿假设。实时未发现R191或更高任务，服务器发布库干净HEAD 27a1c62de66f2d17ba32862d2ef307bc64c2790a，可用18480435200bytes。统一最好仍R15718/24，未称改善或安全。

## 固定方法及数据边界

当前native34 gyro0:3、全部14关节home-relative位置6:20与速度20:34除.05，因果控制0初始读数和固定同相位名义进入私有canonical-root计算。b=RNE(q,v,0)-RNE(q,0,0)，e=(b-bnom)-(bi-bnom0)，十腿raw=+e/kp，原kp13.37；head新增0。仅kinematics/comPos/jacSite/comVel/RNE，不forward、积分、contact或实际力查询。不读actualroot、up、case/seed/path/labels、未来观测、teacher或lookup。固定名义不是本回合未来实际读数。

这是同状态未clip servo代数前馈，不是电机20ms响应、重力/Mqacc/接触耦合模型或安全/耗散证明。R190中head速度不影响腿偏置的分支限制保留；原19项18通过1失败完整证据不重跑、b测试修正不能说成改变kernel。没有pinv、幅值、方向、时间窗、模型参数或学习器搜索。

冻结R157实际初始native50选择器、R122snapshot/节点、原IMU与右髋反馈；新增在原postIMU及右髋double目标之后合并。旧原IMU调整前double request未单独保存的来源限制不回填。本轮保存exact原/new double目标、raw/e/四组b、current moving/static RNE、私有重建速度及原计划差；这些是信号，不是独立决定审计或动态反事实。

## 接口与完整动态门禁

新回归验证worker冻结初始化、因果签名、529实际名义零/control0零、head原样、原对象、14合计reference±.18/joint/5.24rad/s乘.02slew、原规划顺序、禁止private动力学、模型完整MJB和XML include/mesh资产、启动全部已导入Python源码及独立门禁拒绝物理无效。回归不构造动态环境，不把合成计划断言当step_target验收。

动态wrapper逐控制核对native历史prev float32编码、计划与原step_target applied及prev精确相同。原constructor40prepare和显式case reset40prepare分别记录局部批次，异常留存真实完成批次及最新partial/failed causal inputs，不增加prepare积分、等待、沉降或中途root重置。主进程、各spawn和启动器显式捕获main及全部已导入Python源，另存完整compiled MJB/旧physics/所有模型资产及冻结输入前后hash；不是native binary快照。

先新接口回归，再独立6完整smoke：disabled标准/769000/773004，enabled标准/769002/773004。禁用全部原数组/prepare/initHash/peaks逐位R157；enabled标准全程原样、raw/e精确零，所有smoke物理有效。子进程自然退出后重复新回归、实时无竞争任务，才正式同6全部新旧数组逐位，并25baseline+25candidate，共56正式。总62原路径尝试不是62独立起点，原2279最大45.58秒、12秒进入/严格连续30秒/逐500Hz有效，无效可早退且全部保留；smoke无效不得绕过。

## 预算 保存与后续

seed291仅原初始化与记录，不进kernel。learned_parameters=0/search_trials=0。6CPU，OMP/OPENBLAS/MKL各1、CUDA空、JAXcpu，现有.venv和MuJoCo3.12.0；存储2GB，启动至少12GB且保留10GB，不清理他人或未确认数据。实际启动器diagnostics.launch_getup_velocity_bias_r191 --launch，子probe_getup_velocity_bias_r191 --smoke自然退出后同模块无参数正式。

输出ROOT/outputs/getup_velocity_bias_r191_smoke_20261010、getup_velocity_bias_r191_left_20261010及log；orchestration为getup_velocity_bias_launcher_r191_20261010。全部旧源码/失败/terminal/docs不可覆盖，新目录不可重用。保护live_server2664890，不连接、使能或部署真机，原walker/physics/reward/验收保持。

没有标准保留、原24>=22且优于18、全部有效之前，不跳扩大316或未见320。316>=36/40且优于fixedzero/R102全部有效后才冻结320两组20各18/20及原12秒/连续30秒/500Hz；后续更大独立扰动、sensor/executiondelay及右侧/俯卧/仰卧各20门槛不变。320仍未读执行，全部仿真验收前fulltask/hardware/qualification false，ACTIVE每30分钟续跑。
