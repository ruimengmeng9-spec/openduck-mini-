# R193 全恢复段共享波形教师有限计划

2026年10月10日。R192只读核验完成且发布提交为cc3687ee8c877b348ac04f146d9733bec44f8e9a；R191速度偏置候选10/24，统一最好仍R15718/24。R193检验一个新假设：在全部原开发倒地上共同复测同一全恢复段教师，能否避免个别救回伴随更多退化。它不是已识别的唯一根因或已证明改善。

## 共同教师与旧结构的区别

R119只在三状态组上搜索平滑节点，训练还使用短尾标签，扩大时标准失败。R193不修改R122节点或R157选择器，不复用旧节点参数；新增十腿目标波形由四个覆盖整个529步恢复段的全局正弦基函数表示，每腿四个系数，共40个。每个实际教师从标准及全部原24真实完整倒地分别初始化、执行原完整路径，不以短尾、单案例成功或成功并集筛选教师。

这不是从传感器精确拟合动作的静态插值、MLP、BC或案例选择器，也不是R175固定递归读出、R181策略梯度网络/噪声预算或瞬时phase/脚几何/momentum/velocitybias参数扩展。没有新动力学模型或在线教师oracle。波形时间相关，但不改变原reference相位、节点、等待时间或控制频率。有限搜索结束后不追加同波形的幅值、模式数、方向或时间窗预算。

令k为原控制编号，B_n(k)=sin(pi*n*k/528)，n=1..4；两个端点采用其数学零值。波形w=C B(k)，C为10x4、单位rad。原native34当前与冻结同相位名义差，再减控制0因果初始对应差，以gyro1rad/s、up .05、位置.05rad、native速度.05等固定尺度归一化，得到delta。固定标量a=tanh(RMS(delta))；新增十腿request=a*w，四头新增零。这个标量只保持标准和控制0的实际scalar精确零，不搜索gate，也不辨认案例或宣称安全。

只使用当前和控制0的native34、同相位固定名义、原控制编号、固定共享系数和模型关节映射。up通道保持原framezaxis语义，不冒称bodygravity。没有root qpos/qvel、根速度、case/seed/path/label/teacher、未来本回合观测或精确上下文查表输入。seed293只生成一次离线提案和原初始化记录，不进入request函数。

## 完整复测与离线选择

一次固定提案集为零及两组独立10x4正态方向的正负配对，共5个程序。生成标准差1e-5rad，每个系数限1e-4rad，RNG生成后保存；没有自适应均值、CEM代数、网络训练或追加搜索。新增目标最坏代数界4e-4rad不说明接触安全。

5个教师各完整标准+24，125次训练路径尝试。零teacher必须原字段/prepare/initHash/peaks逐位R157。每个非零teacher的初始hash与零配对。离线使用原完整success/valid和原return_sum：候选必须标准保留、所有原物理有效、保留零教师的所有原成功且成功数严格增加，才可能入选；合格者按成功数、最差原完整回报、总回报顺序选择。无合格非零时选零，不按case在线切换，不拼成功并集。这个约束只针对已开发有限集合，不是鲁棒性定理或未见资格。

最终选中teacher及零baseline另各25完整复测，配对hash。正式还含6启动一致性，共181实际最大动态尝试；独立6合计187，不是187独立起点。每次原最大2279控制步45.58秒，物理无效可早退并保存，不称全部走满。全部失败包括退化保留。没有teacher动作MSE或roottruth监督，qpos/qvel仅按原保存及隔离全字段一致性读取。

## 因果接口和原安全门禁

冻结R157真实初始native50选择器、R122snapshot节点、原IMU及右髋反馈。postIMU/right exactdouble之后merge十腿，head原样；合计14路reference±.18，再原joint、5.24rad/s*.02slew及torque。零request保留原target对象，控制0首target逐位R157，标准全程原样、529以后home新增零但原feedback保留。没有额外prepare积分、等待沉降或中途rootreset，原walker/physics/mesh/flags/reward/验收不改。

21项新接口回归覆盖529 actualnominal zero/因果初始zero、全局波形单位/边界/非法输入/正负RNG、保留约束拒绝退化和invalid/pairedhash、worker snapshot加载、原joint/slew/head/零对象、native真实previous编码、两批原40prepare与异常partial、完整XMLinclude/assets/compiledMJB、全部importedPython explicitmain启动捕获、禁止新模块MjData/forward/kinematics/inverse/RNE/积分/contact/force查询。实际测试结果以日志为准，当前源码准备不预称通过。

独立6完整smoke为零标准/769000/773004及固定第一非零teacher标准/769002/773004。零全部原字段prepare/initHash/peaks逐位R157；非零标准也全程原样/request和delta精确零，全部smoke原物理有效才过门禁。不补称两扰动成功为统一成绩。独立自然退出，重跑新回归、实时无竞争任务，才正式同6全部新旧数组逐位比对；不得绕过smokeinvalid。

保存constructor40与case reset40各真实prepare/time、异常完成批次与最新partial、原labels/peaks/hash、delta/activation/wave/request、原postIMU/right double输入边界、原新double目标、base/adjusted/planned/applied/递推prev、所有闭合程序/RNG/选择约束、全执行源与全模型资产/MJB前后hash。原IMU之前request/double未单独保存，不冒称完整重建原IMU。所有旧source/partial来源缺口不回填；计划与实际目标一致不是动态反事实或安全证明。

## 资源和完整验收

启动前实测/data17605296128bytes，主库clean cc3687ee，仅live_server2664890，无R193+实际路径。新live/归档/Git合计预算2GiB、保留至少10GiB，启动至少12GiB；每组前实时空间检查，留出后续归档容量，不清理他人或未确认数据。6CPU、OMP/OPENBLAS/MKL1、CUDA空/JAXcpu、现有.venv。新源码、输出、文档不可重用覆盖；失败使用新记录不改旧事实。

实际新入口diagnostics.launch_getup_shared_horizon_teacher_r193 --launch，后台同无参数。子diagnostics.train_getup_shared_horizon_teacher_r193 --smoke独立自然exit后同无参数formal。输出getup_shared_horizon_teacher_r193_smoke_20261010、getup_shared_horizon_teacher_r193_left_20261010及launcher对应唯一目录和log。

原开发标准保留、24至少22且优于18/allvalid后才冻结316至少36/40且优于fixedzero/R102/allvalid，再未读320两组20各18/20/12秒进入/连续strict30/逐500Hzvalid以及后续更大扰动、sensor/executiondelay、右侧/俯卧/仰卧各20门槛。320仍不读取执行。ACTIVE30分钟，全仿真验收未过不完成；硬件不连接/使能/部署，保护live_server。

