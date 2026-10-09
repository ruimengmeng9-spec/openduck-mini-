# R190 速度相关惯性偏置计算与离线验证

2026年10月10日，R190首次19项离线回归18项通过、1项失败；R190b修正了对head与腿分支耦合的错误测试假设，20项回归全部通过并自然退出。计算内核和补偿公式未改变。两版没有构造恢复回合、执行新控制器、训练、扩大或资格；统一最佳仍R15718/24，全部仿真验收尚未通过，任务保持ACTIVE，真机未连接。

## 新假设与原方法区别

R189独立重算了R188b全部62条轨迹，未发现角动量分配或限位决定不一致，但原候选只有5/24、物理无效3，对冻结R157基线18/24没有改善。R185局部进度、R183接触几何、R179脚姿态及R181完整回合策略梯度也没有解决跨起点保留。这些结果不识别唯一根因，也不证明全部反馈无效。

新的有限假设是补偿当前关节运动与gyro产生的模型速度相关惯性偏置，而不是把角动量误差投影成虚拟速度。它使用固定模型的质量和惯量得到Coriolis与centrifugal项，再由实际原位置伺服系数换算十腿目标请求。它不是旧线性速度gain的幅值扩展、脚位姿或相位反馈、pinv分配、学习器或静态插值。没有参数拟合、噪声、幅值、符号、轴或时间窗搜索，learned_parameters和search_trials均0。

原upvector实际来自imu site的framezaxis，是site的z轴在世界坐标中的方向，不等于机身坐标中的重力方向。新计算不使用upvector，不据此恢复完整姿态或重力方向；仅计算两次同姿态递推之差以消去模型重力项。

## 因果输入与固定公式

函数仅接收current、nominal、initial、initial_nominal四组native34、私有模型对象、原整数控制编号及固定实验启用开关。使用gyro0:3、14个home-relative位置6:20及速度20:34除0.05恢复rad/s。case、seed、路径、labels、实际root位置速度、未来实际读数、接触、教师或lookup都不是推理输入。固定同相位名义来自既有冻结传感器轨迹，不是本回合未来观测；只解码observations[:529,:34]，不解码实际root轨迹qpos/qvel。

私有MjData每次设canonical root位置0、单位四元数、根平移速度0，全部14关节位置速度来自native；gyro经过实际site旋转和angularJacobian解出根角速度坐标。gyro与trunk body不同但同属原刚性weld。只调用mj_kinematics、mj_comPos、mj_jacSite、mj_comVel和mj_rne(flg_acc=0)，没有environment、forward、inverse、积分、contact、distance、passive或实际接触力查询，没有向原模拟器写力或改模型参数。

令b(q,v)=RNE(q,v,qacc=0)-RNE(q,0,qacc=0)，每次两项共享姿态、原重力、模型和flags，不改gravity或关闭任何物理标志。b为速度相关的广义惯性偏置，腿关节单位N·m，不包含重力、M*qacc、约束接触、实际根线速度或电机响应。该定义依据[MuJoCo API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-rne)与实际3.12.0的[递推实现](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_core_smooth.c)，不据此支持恢复或安全结论。

e=(b_current-b_nominal)-(b_initial-b_nominal_initial)。十腿raw=e/kp，head新增0。实际14个servo都是unit gear、fixed gain、affine position bias、无actuator内部动态，kp均13.37 N·m/rad，bias位置项为-kp。正号由同状态、未clip的原servo代数导出：增加raw可增加kp*raw的请求力矩；它不是修改servo gain或直接力矩控制。原joint/reference/slew和torque clip之后的实际作用不保证等于e，更不是20ms动力学预测、耗散、接触安全或未来恢复证明。

control0当前/初始差相消，原529帧actual名义scalar的raw/e精确零；禁用及home529以后信号零。未来包装器仍冻结R157实际初始native50选择器、R122snapshot/节点/原IMU与右髋反馈，零请求保留原目标对象。14原新参考合计各±.18rad，再原joint/5.24rad/s乘.02slew/torque，原50Hz控制、500Hz物理、mesh/flags/reward/验收不变，不加prepare、等待、沉降或中途root重置。

## 首次失败与分支限制

R19019项0.926秒，18通过、1失败。错误测试预期改变head速度会改变腿偏置，实际十腿差精确0。这里根运动被规定、qacc=0，head与腿是不同分支；head影响自身偏置及根反作用，但没有通过未知根加速度或接触耦合到腿的本项偏置。不能把本方法说成全机惯量协调，也不能把head状态被解析说成其必然影响新增腿目标。

R190b保留同一不可变kernel，只替换这项错误测试：验证腿偏置逐位不变、head自身偏置和根角向反作用确有变化、head新增0；增加第20项原kernel SHA256一致性回归。原失败19项、源码、executed_sources、test执行副本、完整日志和results均保留，不覆盖或重跑第一版，不放宽物理或控制参数。

该固定合成样例的腿差0、head差0.04873074184913162 N·m、根角向反作用差0.04691990055038358 N·m，只是离线模型分支关系，不是实际机器人反力测量、原完整倒地root推断或安全结论。

## 闭合验证与来源

R190b20项0.866秒通过：独立12姿态逐部件空间惯量与速度递推、静止精确零/重力相减、二次速度缩放、速度反号偶性、gyro site与weld、native速度单位、529名义零、因果初始零、未用up、head分支限制、原servo正号、home/禁用、原merge边界和零对象、禁止私有forward/contact/积分、完整模型不变、统一合成旋转/平移不变、非法输入与不可变kernel hash。合成数学样例固定seed290，只用于离线回归，不是策略输入或学习器探索。

独立Python逐部件递推从质量/惯量/部件位置和kinematic comVel字段组装空间惯量，不调用mj_rne或已保存偏置作决定；最大差1.6306400674181987e-16 N·m，预定atol1e-14、rtol3e-12。C与Python算术不称逐位；名义零、原对象和原kernel hash是精确断言。统一旋转/平移仅一个合成姿态样例，不是完整全姿态证明。

两主进程与各测试子进程显式捕获main及启动时全部已导入Python源码、路径消歧副本和hash；不是第三方native binary快照。全部XML include/file assets、原参考/stand模型、冻结selector/snapshot和名义压缩输入hash前后相同。完整compiled MJB为53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3，两轮各自前后及原固定模型一致。R190b还核对原失败目录全部bytes前后不变。旧R183/R185/R187/R189来源与partial缺口不回填。

kernel SHA256为44d1f4881ac612e66c2a99fa7bceab16ce0fd4657f3d6bb2089077db3f2d07fa；原main bf9ebe69d85c4a80c6acd7179a61449ab0ece0449f79a60d0f9b0d3e8800544e，新b main 19c29de127342682fcdfcd54da95e07f2f4ed17729af916feb561807e2429c37，本机与服务器实际及执行副本核对。

## 命令 预算与下一验证

实际离线命令为现有.venv/bin/python -u -m diagnostics.validate_velocity_bias_kernel_r190，再同验证器后缀r190b，无额外参数。项目/data/shijinsheng/open_duck/projects/Open_Duck_Playground，ROOT=/data/shijinsheng/open_duck，OMP/OPENBLAS/MKL各1、CUDA空、JAXcpu、MuJoCo3.12.0。输出ROOT/outputs/getup_velocity_bias_kernel_r190_20261010及getup_velocity_bias_kernel_r190b_20261010，日志均目录内regression.log。实测22049403与22266419 bytes，离线预算.25GB、至少保留10GB，无数据清理。

这是离线kernel验证，不是独立完整动态smoke、R157全轨迹parity、新step_target或已执行target验收。两轮new_dynamic_attempts=0/new_controller_run=false。全部仿真门槛未过，不把本次离线通过当18/24提高。

下一轮先实时核对R191或更高source/output/完整命令、资源、最新main和clean。无任务才新唯一动态包装器复用immutable R190 kernel，验证original scalar、原初始化、全部目标对象/总界、规划与原applied/prev逐位、完整模型/资产hash、主进程和spawn导入源、真实prepare异常partial。不得再扩本公式、幅值、方向、时窗或重跑旧方法。

固定后续动态预算为独立6完整smoke（disabled标准/769000/773004，enabled标准/769002/773004），自然退出且全部原字段/prepare/initHash/peaks逐位R157、enabled标准原样和物理有效，回归重跑通过且无竞争任务，才同6启动逐位及25baseline+25candidate共56正式，总62原路径尝试而非独立起点。每次原最多2279控制45.58秒、12秒进入/严格连续30秒/500Hz有效，无效可早退并保存所有失败，不绕过冒烟无效。存储2GB、保留10GB、启动至少12GB；没有参数学习或追加搜索。

当前18/24不允许跳316或未读320。原24>=22且优于18/标准保留/全部有效后，才冻结316>=36/40且优于fixedzero/R102全部有效，再冻结320四十配对起点两个20组各18/20及原12秒/连续30秒/500Hz；后续独立更大扰动/传感器与执行延迟/右侧/俯卧/仰卧门槛保持。320未读取执行，保护live_server2664890，无硬件连接、使能或部署，原walker不改。

新唯一归档只追加R190 failure_snapshot_01及b terminal_snapshot、执行源、前后hash、20回归/完整失败日志/本记录和新scripts，不复制旧动态轨迹。准备时干净发布基线及独立远端均5626d17edc5a3d45a7057f051575cfc78edd9558。新提交须等待上传exit0、独立只读remote main/local clean及GitHub明确提交完整记录核对后才称已发布；严格官方host key、命令级代理、repo-only key不输出/下载/写证据，不force，不改预算。条件的历史文字保留，最终收据补充发布状态。
