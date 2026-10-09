# R188 整机角动量分配完整仿真计划

R187b 的固定角动量内核只通过离线回归，没有验证动态控制或原完整轨迹一致性。R188 接入该不可变内核，验证当前陀螺仪及全部关节位置速度产生的十腿协调请求是否在原完整起身流程中改善结果。当前统一最好仍 R157 18/24，全部验收尚未通过。

## 固定规则与输入

复用 R187b SHA256 fe458c4d12fa48756197d9b500dda688bd9bc4ce441db0e15f6bea7476646ac6。当前 native34、控制零的因果初始 native34 和固定同相位名义传感器进入私有 canonical-root 运动学；实际 root、标签、case、seed、目录和未来本回合观测不进入控制。全14关节参与惯量，新增请求只作用十腿，head 不变。

误差 e=(h-hnom)-(hinitial-hnom_initial)，十腿请求为 -.02*pinv(A,rcond=1e-10)*e。它是瞬时运动学虚拟速度分配，不是20ms电机动力学、力冲量或安全模型；代数残差降低不能证明起身恢复、接触安全或耗散。不扩 pinv、幅值、方向、阈值或时间窗预算。

冻结 R157 初始选择器、R122 snapshot、节点和原右髋及 IMU 反馈。原新增合计相对 reference ±.18，再原 joint/slew/torque；控制零、标准实际 scalar 新增精确零，home529以后原反馈。原50/500Hz、mesh、flags、reward和验收不变，不增加prepare、等待、沉降或中途root重置。

## 有限动态与存储预算

新回归覆盖固定内核及包装器接口、冻结 worker 初始化、head 和总界、准备阶段记录调用顺序与异常 partial、完整 compiled MJB hash 和 XML include/资产文件 hash、全部已导入 Python 源码启动捕获。旧 R187 失败、R185 prepare 缺口及 R183 来源缺口不回填。

先独立6完整smoke：禁用标准/769000/773004，启用标准/769002/773004。禁用全部原数组、prepare、initHash、peaks逐位R157；启用标准原完整轨迹逐位、新增零、所有smoke物理有效。独立自然退出且重跑回归、无其他任务，才正式同6逐位，加25 baseline与25 candidate，共56正式、62含独立smoke的完整路径尝试，不是62独立起点。原最大2279控制步45.58秒，无效可早退并保留；smoke物理无效不得绕过启动正式。

seed288只原初始化和记录，learned_parameters=0、search_trials=0。预算2GB，保留10GB，启动需要至少12GB；2026-10-09 21:30 UTC核对可用19392491520bytes。不得清理别人或未确认数据。输出 getup_centroidal_momentum_r188_smoke_20261010 与 getup_centroidal_momentum_r188_left_20261010 为唯一目录。

## 证据与验收

保存真实失败、可获得的prepare partial、全部轨迹、initHash、原peaks、原/new double请求、同状态pre/post计划差、实际h/A/e/raw/代数残差、固定契约及RNG。qpos/qvel只保留原记录与孤立旧字段一致性，不作控制输入。模型compiled MJB hash覆盖实际编译模型，include及资产另存路径hash；执行源与输入前后核对。

原开发标准保留、24>=22且优于18、全部有效才冻结扩大316>=36/40，再未见320两组20各>=18/20及后续延迟/姿态门槛。320仍不读取执行。smoke、数学实现、小动作或cap不能替代原500Hz接触有效和连续严格30秒验收。任务ACTIVE；真机不连接、使能或部署。
