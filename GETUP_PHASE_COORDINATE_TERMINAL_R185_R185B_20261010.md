# R185 R185b 局部进度坐标完整复测终态

R185第一版13项回归通过后，独立冒烟因冻结snapshot初始化遗漏退出。R185b补充加载与独立初始化回归，14项首次和正式前回归均通过；独立6条完整smoke、正式6条逐位一致性及baseline/candidate各25原完整开发回合已保存闭合。候选4/24，基线18/24，标准保留，候选物理无效3；原开发门槛false。没有改善、扩大或资格，当前统一最好仍R15718/24，仿真任务ACTIVE，真机未连接或部署。

## 固定假设与执行范围

方法区别、因果输入和预定预算见GETUP_PHASE_COORDINATE_PLAN_R185_20261010.md。当前与因果初始14关节位置相对同相位名义的误差变化，投影到冻结名义关节运动对控制步的局部切线，以固定.02rad平方正则得到进度坐标，clip至±1控制步。沿原reference曲线线性插值，仅移动reference贡献；原IMU、右髋反馈及snapshot节点仍按原控制编号计算。不是重放整个反馈器、最近邻名义状态搜索或加等待。learned_parameters=0/search_trials=0，无学习器拟合和参数搜索，不追加同坐标/正则/offset/方向/window预算。

实际命令是现有.venv/bin/python -u -m diagnostics.probe_getup_phase_coordinate_r185b --smoke，然后同模块无参数。启动器diagnostics.launch_getup_phase_coordinate_r185b --launch，OMP/OPENBLAS/MKL各1、CUDA空/JAXcpu。项目/data/shijinsheng/open_duck/projects/Open_Duck_Playground，seed285仅实验记录及原初始化接口，不进入策略。独立目录getup_phase_coordinate_r185b_smoke_20261010，正式getup_phase_coordinate_r185b_left_20261010，均在outputs下及同名.log。

控制0 d/坐标/新增零，第一target逐位R157；标准actual scalar精确零，禁用保留原对象；home529步后原反馈、信号零。全部14原新相对reference合计±.18，原joint/torque/slew/50Hz/500Hz/mesh/flags/physics/reward/验收固定。没有case/seed/path/label/root真值/未来actual输入或精确上下文lookup，只有预存参考序列的局部切线和插值。无额外prepare积分/等待/沉降/中途root重置。

## 初始化失败与来源保留

R18513项回归.067秒通过，首zero_parity三个job投递后，在env.reset及选择器准备计算之后、调用scalar_program时local.WEIGHTS仍None，触发TypeError。这是worker未加载frozen/snapshot.npz，不是策略或物理验收失败。回归未覆盖worker完整加载，原源码、executed_sources、冻结文件、契约、日志及目录原样保存；没有完整case result/trajectory或部分prepare帧落盘，不能补造3条闭合回放。env.reset的原prepare过程已调用，不声称没有任何物理积分；未到恢复step_target控制入口。该失败目录不计入62闭合动态尝试。

R185b仅新增snapshot读取与feature_mode断言，并新增初始化回归，控制kernel、正则、offset范围、物理和验收不变。14项首次与正式前各.078秒通过。旧R185 main SHA256618f85f7a361cd78dc76970f976ce154b8100fb8d1d6875f07b691a44d9516ad，修正版62109cd963e4ad406dfc69adaf9cb0565ab39c17358804b11ff4f79a1a552842，两端实际及各executed_sources副本一致。两版main均显式保存启动源码，不重复旧R183遗漏main的缺口；原初始化异常位于回合记录try之前，已有prepare缺失如实保留。

## 独立冒烟与正式结果

独立禁用标准/769000/773004所有原字段、prepare/initHash/peaks逐位R157；启用标准也逐位，11.08秒进入/34.70秒严格尾段/新增零。启用769002完整成功有效，11.10/34.68；773004完整失败有效。两例最大未合并reference插值请求.929399974496804/.412362744876843rad，坐标最大1步。它们不是实际新增动作或总修正，不能用一个控制步坐标界称动作很小或安全，单例成功也不是统一成绩。

独立smoke自然退出后，安全启动器重跑14项回归才正式；同6条全部新旧数组/hash/peaks逐位独立smoke。正式56=6startup+25baseline+25candidate，另独立6共62闭合动态尝试，不是62独立起点。每次原最大2279控制步45.58秒、12秒进入/连续严格30秒/逐500Hz有效；3个candidate物理无效提前终止，不能称62全部走满。其余59均保存2279控制步。training_closed保存固定契约与seed RNG，不是训练梯度或新学习器；结果保存后的Python自然退出状态以实际launcher日志和完整命令核对为准，不发送信号。

25baseline所有原字段/peaks逐位R157，candidate/baseline初始hash配对。候选救回769002/773005，退化769001/769003/769005/769006/769007/773002/773003/773006/773007/773008/773009/773010/773011/773012/773013/773014。保留原成功769000/773000，合计4/24。24扰动首次非零同状态post-slew直接差均控制1；成败和物理峰值全部沿用原回合，不几何重标。

773005未合并reference请求峰值.5238911467021846rad，同状态pre-slew与post-slew直接差均.18515356382105191rad。769002请求.929399974496804，合并后pre-slew直接差.25505045947144034，post-slew直接差.18502845160690617。两路目标之差可超过.18，因为原新各自受相对reference总界约束；它不是新目标相对reference超过.18的证据。多数访问回合的combined峰值.18附近，原double浮点峰值.18000000000000005保持原1e-12算术断言，不改物理上限。这些保存差不是传递增益、未来预测或动态反事实。

## 三个原物理无效回合

769006保存45控制记录，原self峰值.030031150307319416m，raw request/pre-slew直接差.01634004263028979rad，post-slew直接差.0105546676352552，总修正.031436336179644364。773010保存44控制记录，self.029953693111239726m，raw/pre-slew.024478087558393348，post-slew.01581134666360018，总.04854491238035896。773011保存45控制记录，self.029970061548432375m，raw/pre-slew.013157747762945565，post-slew.010532040138994336，总.04362033723395309。

这些例子未到总.18界便物理无效，不能用单一cap饱和解释全部失败。保存控制数量/末端不是首次500Hz越限子步，极小或较小动作也不证明安全。原全部轨迹、失败和peaks保留，既不简单补偿阈值、也不从不同访问轨迹的sensor分叉认定唯一原因。局部进度假设被本次完整结果拒绝为改善策略，但不证明所有相位控制无用。

## 唯一追加与后续

新archive_getup_phase_coordinate_terminal_r185_r185b只在自然退出、无更高任务/活跃publisher、最新main和clean核对后执行，追加原R185 failure_snapshot和b独立6/正式56 terminal_snapshot、全部实际失败/原标签/peaks、14回归/源码/固定RNG及新记录。归档再单独核对25原全字段配对、6formal/smoke全数组逐位、源及复制目录hash前后一致。qpos/qvel仅此孤立全字段equality，不进入phase控制。保存请求/合并前后同状态差统计不等于独立重算phase/规划决定，后续R186优先只读重算。

初始独立核对main及干净服务器HEAD35481a7d2a8fdfe04d2cdfc6048a716a85f2ec2e。新提交和成功上传以实际收据及之后独立远端核对为准，不本地提交即称已发布；旧所有执行源/失败/startup/terminal/docs/bundle不可覆盖。当前约91MB正式+12MB独立+1.7MB原失败，在2GB含快照预算内、/data仍约19GB。保护live_server，不清理未确认数据，不导出key。

下一轮先检查R186或更高实际任务，独立只读smoke自然退出后才正式审计这62既有轨迹的实际phase位置差/名义切线/坐标/插值请求/原右髋反馈和原参考cap/joint/slew决定，沿声明的保存double原目标输入边界，不假称原double IMU已重建；不重跑动态实验或加本方法预算。保持原标签与源码/evidencehash、保留初始化失败缺失。之后才提出真正不同有限方法，当前18/24不能跳316或320。320未读执行，所有原验收门槛不变，全部仿真通过后才结束续跑，不部署硬件。
