# R188 整机角动量分配完整复测终态

固定角动量分配 R188b 的独立6例冒烟与正式56次完整路径复测均已保存闭合结果。候选5/24、标准成功、物理无效3，冻结 R157 基线18/24、标准成功、无效0；gate=false，没有改善、扩大或资格，统一最好仍R15718/24。结果保存后的进程自然退出须另核对，不能把结果文件当退出。

## 固定方法与实际执行

R188b复用不可变 R187b kernel SHA256 fe458c4d12fa48756197d9b500dda688bd9bc4ce441db0e15f6bea7476646ac6。当前native34的gyro、全部14关节home-relative位置与/.05恢复的速度，因果初始native34和固定同相位名义形成整机trunk坐标绕质心角动量。私有canonical-root仅kinematics/comPos/angmomMat/jacSite；实际root、upvector、case/seed/path/label/未来观测不作新增控制输入。head参与惯量但新增仅十腿。

e=(h-hnom)-(hinitial-hnom_initial)，raw腿request=-.02*pinv(A,rcond=1e-10)*e。它是瞬时运动学虚拟速度分配，不是电机20ms效果、动力学预测、接触安全或耗散证明。原R157初始选择器、R122 snapshot/节点及原IMU、右髋反馈冻结；14原新相对reference合计±.18，再原joint/5.24rad/s/.02slew/torque。head保持原target，control0原对象、标准新增精确零；home529后原反馈，不重置root。

实际独立命令为现有.venv/bin/python -u -m diagnostics.probe_getup_centroidal_momentum_r188b --smoke，正式同模块无参数。OMP/OPENBLAS/MKL各1，CUDA空、JAXcpu。seed288仅原初始化/记录，learned_parameters=0/search_trials=0，无参数拟合或搜索。原50/500Hz、mesh/flags、physics、reward与12秒进入、连续严格30秒验收不变；无额外prepare、等待、沉降或恢复中root重置。

## 两个实现异常及修正边界

R188第一版24项回归2.258秒通过；独立zero_parity三job在准备记录检查异常，恢复controls_completed均0，没有闭合恢复轨迹。每job保存80个真实prepare帧及时间：原环境构造会调用一次40步prepare，显式case reset再调用40步；首版记录器错误累加两次帧后检查40。旧日志、源码、partial_trajectory和failed_causal_inputs全部保留，不能称无任何积分，也不补造三闭合回放。

R188b仅将每次record_preparation调用分为独立局部批次，保留最新partial及已完成批次，新增连续两次prepare回归；25项1.458秒通过。原额外构造prepare本来存在，记录器没有增加积分、observe次数或改变调用顺序。新完整trajectory保存原实际case40帧，并另存construction_preparation_sensors/times；旧R185 prepare缺口不回填。

R188b独立6例已闭合并自然退出后，原启动器在检查report['phase']['physical_failures']时KeyError，因为第二个遗留字段名没改；当时正式未启动。首版启动器不覆盖。新launch_getup_centroidal_momentum_r188c只修正门禁为momentum，复用已保存独立smoke，不重复动态冒烟；重新25项回归1.505秒通过后执行原56正式预算。R188c不是新控制公式，没有额外动态尝试。

R188 probe源码SHA256 f1fc48e61c757fe1ad4228d12d217432eb4f1b6eb290bfa7dba625541272ca63；R188b probe 4a24d4797579fd1549c54424ab80a680677ec90725e1f006bee40f63e734ba94；旧b launcher 7604a3db23fdce9a1d1447efb5b5fd83ae9d26318422910697a742353a7492d8；新c launcher 6bf9bac812700b52707d7cdfcf08cfffb27f6b6d60dc8bf1549abd66d6e3cbf0。本机与服务器对应hash一致。

## 完整预算与一致性

独立6与正式6启动一致性+25baseline+25candidate，共62闭合恢复路径尝试，不是62独立起点；59条走满2279控制步45.58秒，三个candidate物理无效44/304/303步早退，失败不排除。R188首次三个仅prepare异常不计闭合恢复回合；c启动器无新增smoke。

禁用标准/769000/773004全部原字段、prepare/initHash/原peaks逐位R157。启用标准全部原完整字段逐位R157，11.08秒进入、34.70秒严格尾、新增全程0。启用769002成功有效11.12/34.66，raw最大.1152747971556342rad、同状态post-slew差.09687201110183358；773004完整失败有效，raw.26466367019901177、post.17688880831137452。仅已开发smoke，不是统一成绩或安全证明。正式同6全部新旧数组逐位独立；25baseline原全字段与原peaks逐位R157，25candidate/baseline初始hash配对。

主进程、各spawn工作进程及c实际启动器显式捕获启动时所有已导入Python源码，包含__main__、固定kernel和依赖；第三方原生二进制不是Python源码快照。输入hash与全部XML include及file-backed mesh/资产hash前后相同。实际compiled MJB序列化完整模型SHA256 53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3，逐轨迹前后一致；原既有physics hash4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae保持。MuJoCo3.12.0。不将顶层XML单hash冒称完整include指纹。

## 跨案例结果及限制

五个扰动成功为769000/769002/769003/773013/773014；仅救回769002。退化769001/769005/769006/769007/773000/773002/773003/773006/773007/773008/773009/773010/773011/773012，共14例。24扰动首次同状态post-slew非零均控制1，raw最大范围.04368684326591979至.34020143987395807rad；这些是已保存信号，不是独立重算完整momentum/限位决定、动态反事实或唯一根因。

| 无效case | 保存控制数 | 原self峰值m | raw最大rad | pre差最大rad | post差最大rad | 原新总峰值rad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 769001 | 44 | .030004594286001874 | .04368684326591979 | .04368684326591976 | .04368684326591976 | .07125905265760757 |
| 773011 | 304 | .048669140173904564 | .08959922836731452 | .08732832953143357 | .08426752462222553 | .17999999999999994 |
| 773012 | 303 | .048711971758590954 | .09776281371925288 | .09776281371925288 | .0823301082324457 | .12459774324492712 |

其他原物理项未超。控制末端与记录数不是首次500Hz越限子步；未触cap或更小raw不证明安全，不能浅几何重标。即使未限位代数残差最大仅约5.10e-16kg m²/s，仍不代表实际电机达到虚拟速度或随后角动量/恢复改善。不同访问轨迹的sensor/target分叉不是同状态即时动力学效应。

## 保存和后续门槛

唯一新归档应保存R188 failure_snapshot_01、b独立6及正式56 terminal_snapshot、c启动器终态，全部失败、原label/peaks、pairedhash、h/A/e/raw/residual、原/new double请求/同状态pre/post动作差、固定RNG、24/25回归、模型与源码hash、新记录及脚本。qpos/qvel仅原轨迹记录与孤立全字段equality，不作momentum反馈。旧固定失败/source/terminal/docs/scripts不覆盖；本轮存储预算2GB、保留10GB。

先核对自然退出、无更高编号任务及新实际main/clean，再正常快进发布；上传成功收据之后还需独立只读main核对。此段是准备记录时的条件，不预称GitHub已发布。

下一轮若无R189或更高任务，优先新只读审计全部62已有轨迹，从actualnative与因果初始、固定名义独立重算h/A/e/raw/residual、head零、原double边界之后合计reference/joint/slew及applied/prev。保存h/A/e/raw只比较，不代替决定；原IMU完整double前输入若未保存须明确边界，qpos/qvel只孤立equality。允许私有canonical运动学，不环境积分/force/物理重标，先独立只读smoke自然退出再正式。不能重跑R188/b、扩momentum pinv/幅值/方向/窗口或因代数残差小扩大资格。

全部原22/24→31636/40→未见320两组18/20及后续延迟/姿态门槛保持。320仍未读取执行。任务ACTIVE，每30分钟续跑；full_task_completed/hardware_readiness/qualification均false，真机不连接、使能或部署。
