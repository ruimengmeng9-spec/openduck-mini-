# R186 局部进度坐标与原目标限位只读审计

2026年10月10日，R186独立只读冒烟和正式审计均已完成并自然退出。R185b全部62条既有轨迹的关节误差变化、名义切线、进度坐标、参考插值及原目标限位决定逐位重算一致。没有新动态回放、策略训练或标签重标。R185b仍4/24、标准保留、物理无效3，冻结R157基线18/24；当前最好仍18/24，续跑ACTIVE，全部仿真验收及硬件可用性未通过。

## 范围与实际命令

项目/data/shijinsheng/open_duck/projects/Open_Duck_Playground及现有.venv。OMP/OPENBLAS/MKL各1、CUDA_VISIBLE_DEVICES空、JAX_PLATFORMS=cpu。安全启动器diagnostics.launch_getup_phase_coordinate_audit_r186 --launch先8项回归，独立子命令diagnostics.audit_getup_phase_coordinate_terminal_r186 --smoke自然退出后，重跑8项回归才执行同审计模块无参数正式。首次8项.242秒、正式前8项.250秒均通过，无新失败或修正版；launcher已输出natural_exit=true/new_dynamic_trajectories=0。

输出为ROOT/outputs/getup_phase_coordinate_terminal_audit_r186_smoke_20261010和getup_phase_coordinate_terminal_audit_r186_20261010及同名.log，ROOT=/data/shijinsheng/open_duck。独立6条已有轨迹是正式zero标准、phase标准、baseline769000及candidate769002/773005/769006，覆盖非零救回和物理无效早退；不是新的倒地起点或动态尝试。

正式审计原独立6和正式56全部62轨迹，共134595控制记录。其中59条保存2279步，三个原物理无效轨迹分别45/44/45步，未排除失败或称全部走满45.58秒。正式独立同6行、25终态配对及对应信号数组逐位相同。正式保存25candidate及3其他代表共28份信号，独立6份；正式目录13077132bytes、独立2717044bytes，在.25GB审计预算内，保留至少10GB。

## 独立重算与数据边界

标量仅解码明确白名单的actual native55、prepare及double原目标输入/比较字段，不解码实际root qpos/qvel。用当前和控制0的14位置先float32读取再提升double，重算d=(q-qnom)-(q0-qnom0)。从冻结名义actual位置独立求529步中央/端点切线v，按固定(.02rad)^2正则和±1控制步坐标边界重算s，再按原reference序列的端点clip及线性插值重算请求。保存的s、d、v、request、adjusted/planned只作为比较，不作为这些决定的输入。

冻结实际prepare初始选择器重新计算gains/choice/logits，原右髋函数从当前实际float32观测重新计算feedback。原double IMU输入及原IMU调整前目标未单独保存，因此以original_double_pre_slew_target_rad作为明确边界，它已含原IMU/右髋结果，不冒称独立重建全部double IMU或原程序动作。

从该double边界输入、重算request和原reference独立实现原合并公式：总参考修正clip至±.18，再joint范围；从递推真实前目标独立计算原5.24rad/s、.02秒slew。前目标在每控制步与native历史目标的float32 home-relative编码精确一致，控制0使用原home准备结束的home目标，不进行任何准备积分。重算base、adjusted、pre-slew直接差、最后计划和applied全部逐位；post-slew直接差由重算两路计划相减，不用已保存extra替代决定。

控制0/标准/home的新增与坐标精确零，禁用目标对象保留；14路总参考修正的逐轨迹原峰值精确。标准新增零不等于扰动安全。只加载MjModel的actuator映射、home和joint范围以及指定不变数组，没有MjData/环境构造、kinematics、forward、积分、contact/distance或force查询。模型数组、完整输入压缩文件字节与追踪源码/冻结数据SHA256前后不变；读取压缩文件字节用于hash不等于解码root作反馈。

8项回归覆盖随机/端点独立phase与原kernel、529 actual名义scalar零、因果初始零、合并/限位逐位及零对象、无保存决定/root输入、禁止环境/MjData/动力学、62预算和非零/无效冒烟、home/禁用零及非法输入。没有改变旧R185或R185b源码。

## 原标签与实质限制

25candidate/baseline初始hash配对，baseline全部原R157保存字段和peaks逐位一致。只有这一孤立全字段equality才解码qpos/qvel，绝不进入phase或规划。candidate与baseline不同字段如实保存，不要求两者相同。原success/valid及500Hz峰值不重算、不重标。

两个救回769002/773005与16退化保持原记录。三个无效769006/773010/773011的坐标峰值分别.23003109341542788/.2336806889952775/.2293557925793133控制步；总参考修正.031436336179644364/.04854491238035896/.04362033723395309rad，原self峰值.030031150307319416/.029953693111239726/.029970061548432375m。重算与保存完全一致，不支持把它们当限位实现错误或简单cap饱和解释；控制记录末端不是首次500Hz越限子步，较小动作不证明安全。

R185第一版在初始化时调用过原prepare但未保存部分帧的缺口不回填；旧失败、executed_sources和日志前后hash不变，不补造闭合回放。R185b main启动副本仍62109cd963e4ad406dfc69adaf9cb0565ab39c17358804b11ff4f79a1a552842。R186 main显式保存executed_sources，源SHA256 b1aa01b5178054dfb3de96ac73e34e79632e17731844eef36e73459337e8acfb，两端当前和执行副本一致。

一致性审计不能给出动态反事实、传递增益、未来恢复、接触安全或唯一失败根因。未发现坐标/插值/原限位决定不一致，不证明所有相位控制无用，也不授权继续搜索同phase正则/offset/方向/时间窗。

## 唯一追加与下一轮

新archive_getup_phase_coordinate_audit_r186只追加两新terminal_snapshot、62核验/25配对/28正式及6独立信号、源hash/8回归/日志/新执行源及本记录，不再复制旧动态轨迹或重写旧失败、source、doc、归档或bundle。初始独立核对GitHub main和干净服务器均b4a26e7c7f23162027d254d944d9b1c69fed62bf；新提交、完成上传及独立远端核对以实际后续收据为准，不本地归档即称已发布。

服务器可用约19GB(df100%)，保护live_server2664890，不清理他人或未确认数据；repo-only部署密钥不输出/下载/写入证据，严格官方host key和正常快进。没有硬件连接、使能、部署或原walker修改。

下一轮先查R187或更高实际任务、最新main/clean及资源。无任务才结合完整跨案例结果提出真正不同有限方法，先说明假设/区别/因果输入/离线数据用途/存储与动态预算；不得重复R186或仅加方向阈值统计、追加本phase/旧projection/PG/固定读出/脚幅值/旧gain残差mix速度记忆/静态精确插值/旧BC/几何验收窗口预算。先actualscalar因果/名义zero/边界/init/原完整bitwise回归及独立完整smoke自然退出，再正式有限训练，不预称改善。

当前18/24不允许跳扩大或资格。原24>=22且优于18/标准保留/全部有效后才冻结316>=36/40且优于fixedzero/R102全部有效，再未见320共40、两不重叠20组各>=18/20、12秒进入/连续严格30秒/逐500Hz有效；后续独立更大扰动/延迟及右侧/俯卧/仰卧各20门槛保持。316/318已开发，320仍未读执行。全部仿真门槛通过并保存证据/列硬件未通过项后才结束续跑，不自动部署。
