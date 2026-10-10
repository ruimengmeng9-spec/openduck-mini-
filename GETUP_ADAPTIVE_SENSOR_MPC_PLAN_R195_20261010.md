# 在线自适应传感模型起身实验计划

2026年10月10日，用户授权尝试此前提出的新方向。R195不加载或扩展R172和R173的342维ridge模型，不追加R193全局harmonics预算。统一最好仍R15718/24，全部仿真门槛未过。本轮是实验，不预称可辨识、安全或改善。

## 因果模型与动作辨识

native34当前与固定同相位名义之差，减去因果初始差，按gyro1、framezaxis0.05、位置0.05、native速度0.05缩放。framezaxis不是bodygravity。78维输入为当前34偏差、上一次34偏差增量和十腿当前已执行目标减同状态原计划的差，动作按0.0001rad缩放。每回合零权重、100I协方差，遗忘因子0.995；下一真实观测到达后才递推更新。在线无root、qpos/qvel、case、label、path、未来观测或lookup，无私有模拟器或几何查询。

seed295仅生成共同实验创新和一次探针，不作传感特征。十腿独立normal标准差0.00001rad乘tanh因果偏差RMS，clip每腿0.0001rad。正负程序仅控制64加减一次0.00001rad共同单位探针，之前真实全部历史逐位相同；控制65下一传感器差与控制64模型预测动作差比较。随后分叉不当即时反事实。全部失败和无效早退保留，不排除其记录或重标物理。

独立smoke和正式24扰动成对测试分别要求gyro、framezaxis、q、native速度四组pooled平方误差各小于零效应误差的0.8，前缀精确相同、实际动作差非零、标准新增零、校准全部原物理有效和无数值fallback。任一不满足则不运行候选、不追加预算。这不证明一般可辨识性、未来多步精度或安全。

## 预测控制和有限尝试

候选控制0..63沿校准规则，64..528逐步使用本回合递推模型。固定3控制步即60ms、十腿常值动作；平方传感偏差代价加0.001动作平方，一次10维线性求解后clip动作0.0001rad。预测是数学传感模型，不是MuJoCo未来回放或稳健安全保证，不冒称预测中未来限位和接触安全。数值异常记录后回到原计划，不用fallback绕过模型门禁。home529新增清零，原feedback保持。

独立8次为zero标准/769000/773004三次，校准标准一次，正负校准769002/773004四次；每条真实完整原路径，无效可早退。回归通过、独立自然exit和全部smoke门禁后才formal。formal重复5次启用校准轨迹全数组逐位，再zero/正校准/负校准各标准加原24共75条，共80条；zero三项在完整25baseline内核对，避免多占重复预算。辨识formal成对效应门禁通过且自然exit后，另起独立3条候选完整smoke（标准/769002/773004），自然exit及回归通过后才正式共同25候选并核对相同3条新旧数组。这28条留给后续独立进程阶段，本R195辨识入口不偷偷启用MPC动态。辨识8加80及候选3加25合计最多116次，每次最多2279控制45.58s；不是116独立起点。无方向、幅值、轮数、窗口或阈值追加。

只在真实到达的本回合历史上在线学习；下一观测仅作已发生转移监督。旧数据只用于冻结名义及原zero逐位核对，不拟合旧模型、teacher动作、成功标签或静态MSE。保存每步因果输入、实际执行差、更新残差、预测和终态权重/协方差/RNG供复算。

## 存储接口与验收

最多6CPU，OMP/OPENBLAS/MKL各1，CUDA空/JAXcpu。live、归档和Git增量合计限2GiB，启动至少12GiB空闲并保留10GiB；不删未授权数据。启动前发布HEAD2d4ff19d92049da32cfd9a9867f588141c2156f8干净，无R195至R249任务/source/output，free20796264448bytes仅当时快照。

冻结R157真实初始native50selector、R122snapshotnodes、原IMU/rightfeedback。postIMU/right exactdouble是明确输入边界；旧before-request和IMUdouble未保存不冒称重建。十腿新增/head原样，14总修正相对reference±0.18rad，再原joint和5.24rad每秒乘0.02slew/torque；zero保留原对象、control0和标准新增零、home原feedback。constructor40与reset40真实prepare/time分开，无额外积分、waitsettle、状态复制或rootreset。完整MJB、全XMLassets、冻结输入、main/worker/test/launcher全部导入Python启动副本及前后hash保存；非native binary快照。旧来源缺口不回填。qpos/qvel仅原记录与隔离旧字段equality，不进入辨识或控制。

全部候选共同Std+24真实完整复测，不成功并集。标准保留、24至少22且优于18/allvalid，才冻结316至少36/40且优于fixedzero/R102/allvalid，再未见320两组20各至少18、原12s进入/连续strict30s/逐500Hz有效，再更大扰动、sensor/executiondelay、right/prone/supine各20门槛。320仍未读执行。保护live_server，不改walker/physics/reward/验收，不接入、使能或部署真机。

[Lu和Cannon的自适应MPC研究](https://arxiv.org/abs/2211.09275)建立在特定线性系统、扰动界和激励假设上；不能将其理论保证直接套用本接触机器人。本轮超参数是固定实验设计。

