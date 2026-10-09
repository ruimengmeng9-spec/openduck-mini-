# R189及R189b 角动量终态只读审计

2026年10月10日，R189b独立只读冒烟与正式审计均完成并自然退出。全部62条R188b既有恢复轨迹、135112条控制记录的角动量和原限位决定重算逐位一致，没有新增动态回放、训练、扩大开发或资格。R188b候选仍为5/24、标准成功、物理无效3、gate false；冻结R157基线18/24且无效0仍是统一最佳，全部仿真验收尚未通过。

## 首次失败与修正范围

R189先执行10项离线回归，9项通过，1项失败，0.887秒后异常退出；未进入已有轨迹审计，也未构造任何动态回合。失败测试误要求质心角动量矩阵根平移列逐位零，实际最大舍入项为1.734723475976807e-17。R187b既有独立部件矩阵核验已采用1e-15算术绝对界，矩阵的理论零不等于浮点逐位零。

新R189b仅将这一离线零列检查改为atol 1e-15、rtol 0，增加根平移列与原C矩阵实现逐位相同的第11项回归。角动量公式、原R187b kernel、控制参数、模型物理、奖励、碰撞和验收没有改变。初始11项0.873秒通过，独立冒烟自然退出后，正式前11项0.765秒再次通过。

R189原三源码、完整失败回归及启动器日志另存failure_snapshot_01，不覆盖或重跑。原回归没有启动时源码复制，归档中的原三源码是退出后核对一致的副本，不称原启动快照。R189b两个审计主进程均显式捕获main和已导入Python源码，包括路径消歧副本和hash；它不是第三方原生二进制的启动快照。R183来源与partial缺口、R185未保存prepare、R187有限源码及顶层XML范围和R188实际prepare失败等旧事实均不回填。

## 实际流程与覆盖

项目为`/data/shijinsheng/open_duck/projects/Open_Duck_Playground`，ROOT为`/data/shijinsheng/open_duck`，使用现有`.venv/bin/python`、MuJoCo 3.12.0，OMP、OPENBLAS、MKL各1，CUDA空，JAX cpu。

实际启动器为`diagnostics.launch_getup_centroidal_momentum_audit_r189b --launch`；先回归，再`-u -m diagnostics.audit_getup_centroidal_momentum_terminal_r189b --smoke`，等待独立子进程自然退出，重跑回归并确认无其他任务后，同审计模块无参数正式。启动器结果natural_exit true、exit_code 0、new_dynamic_trajectories 0。

输出为ROOT下的`outputs/getup_centroidal_momentum_terminal_audit_r189b_smoke_20261010`和`getup_centroidal_momentum_terminal_audit_r189b_20261010`及同名log。独立6条来自正式zero标准、momentum标准、baseline 769000、candidate 769002成功、773004完整失败有效、769001无效44控制；后三覆盖非零请求。它们是已有轨迹的只读核验，不是六次新回放或独立起点。

正式包括R188b独立6及正式56的全部62条。59条原2279控制45.58秒，3条无效分别44、304、303控制，不排除失败、不称全部走满。25终态candidate与baseline及R157的初始hash配对；baseline所有原字段和原物理peaks逐位R157。两轮6行、25配对和6对应全部重算信号数组逐位相同。正式保存25候选加3代表共28信号，独立6信号；不是28或6新轨迹。实测目录分别32490137和18642947 bytes，在0.25GB预算内，至少保留10GB，没有清理他人或未确认数据。

## 独立角动量计算

审计只解码白名单observations、preparation_sensors、original_double_pre_slew_target_rad与比较字段。当前及因果初始native34仅gyro 0:3、home-relative关节位置6:20、速度20:34除0.05参与新计算；up 3:6不用。冻结同相位名义传感器不是本回合未来实际观测。case、seed、目录、标签、真实root、未来子步和精确context lookup不进入推理函数。

独立实现私有canonical root，根位置0、单位四元数、根线速度0，全部14关节含head由当前native重建。只调用mj_kinematics、mj_comPos、mj_angmomMat和mj_jacSite；按实际gyro site旋转及角Jacobian求根角速度坐标，再将质心角动量矩阵转换至trunk坐标。gyro body与trunk body不同但body_weldid相同的映射、native速度单位及十腿列均重新核对。无需动态environment，不forward、积分、contact、distance、force或成败物理重标。

独立重新计算四组h和十腿A，e=(h-hnom)-(hinitial-hnom0)，腿raw=-0.02 pinv(A,rcond 1e-10)e，head新增0，残差e+A(raw腿/0.02)。保存h、A、e、raw、residual只作比较，不代替决定；未调用原kernel的measure或momentum_request计算审计结果。全部62的这些信号逐位一致。名义和control0新增request、error、residual精确零；标准A可以非零，不能把它说成全零矩阵。home529后及禁用信号零。

原R187b kernel SHA256为`fe458c4d12fa48756197d9b500dda688bd9bc4ce441db0e15f6bea7476646ac6`，R188b probe为`4a24d4797579fd1549c54424ab80a680677ec90725e1f006bee40f63e734ba94`，均未修改。新审计主源码SHA256为`658fbfe0dc9799a0a5909a0bf1cb46cef88d7ca506ef97c2b0820e9113ace550`，本机、服务器和执行副本一致。

## 原目标与限位输入边界

冻结选择器由实际prepare末native50重算choice、gains、logits，原右髋函数由当前实际native重算，均逐位一致。`original_double_pre_slew_target_rad`已经包含原IMU及右髋结果，是明确输入边界；其前双精度request和原IMU double没有单独保存，不能声称重建全部原IMU或上游策略目标。

边界之后独立重算十腿new加old相对原reference合计±0.18、joint clip、5.24rad/s与0.02秒slew，head保持fixed原样，zero保留原目标对象。递推previous从home开始，与每控制native历史float32编码精确相符；base、adjusted、pre-slew直接差、planned、实际applied和post-slew两重算计划之差全部逐位，14合计峰值精确。控制0使用原目标，标准全程新增零；不增加prepare、等待、沉降或中途root重置。

实际完整编译MJB指纹`53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3`与62条原字段相同，审计前后相同；XML include和file assets、冻结输入、实际旧导入源码及保存副本、失败证据和压缩轨迹byte hash前后不变。qpos和qvel只在隔离的原全字段equality比较中解码，不作角动量或规划输入；hash完整压缩bytes不等于解码root用于反馈。

## 结果含义与后续约束

769001原44控制的self峰值0.030004594286001874m，773011原304控制0.048669140173904564m，773012原303控制0.048711971758591954m全部保留；raw与原合计界、原全部peaks不重标。记录末端不是首次500Hz crossing。审计未发现瞬时角动量分配和限位决定不一致，但代数残差小不证明电机20ms动力学效果、耗散、接触安全、恢复改善、动态反事实或唯一根因；不同访问状态的随后分叉也不证明同状态即时效果。

不得继续追加本momentum、pinv、幅值、轴、方向或window预算，不重复R188动态、R187 kernel回归或本审计。下一实际编号需先实时核对无更高任务、资源和最新main/clean，再依据完整跨案例结果提出真正不同且有限的全过程因果反馈或多个真实倒地共同完整复测的稳健教师方法；teacher和root真值只可离线监督，不能online oracle或context lookup。旧禁止结构和全部原门槛不变。

候选标准保留、原24至少22且优于18并全部物理有效后，才冻结扩大316至少36/40且优于fixedzero与R102、全部有效；再冻结未见3200000至3200039两个20组各18/20，12秒内进入、连续严格30秒和逐500Hz有效。316与318已开发，320仍未读取执行。之后更大独立扰动、传感器与执行延迟、右侧及俯卧仰卧各20未见18/20都须通过。full_task_completed、hardware_readiness和qualification均false，继续ACTIVE30分钟，不连接、使能或部署真机，保护live_server2664890。

## 唯一归档与发布条件

新归档脚本`archive_getup_centroidal_momentum_audit_r189_r189b.py`保存原失败及两个新terminal_snapshot、完整11回归日志、启动器自然退出收据、62核验、25配对、28正式加6独立信号、源码及模型前后hash；不复制或重归档旧动态轨迹，不改旧固定source、failure、doc、terminal、scripts或bundle。

准备时发布基线为已独立核对的干净`b711a9463dded8439934a305aaa9487d07af61ea`。新唯一正常快进提交与上传须等待helper退出0、独立只读remote main及clean核对，并从GitHub明确提交完整读取本记录后才称发布。新wrapper仅确切git push上传timeout 3600秒；不强推、导出部署key、关闭host校验或改训练预算。保存条件的历史文字不覆盖，最终成功收据和定时快照补充实际发布状态。
