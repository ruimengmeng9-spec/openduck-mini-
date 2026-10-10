# 完整回合总目标 PPO 训练终态

2026年10月10日，R196在用户明确允许替换起身阶段旧反馈后完成有限训练并自然退出。两轮确定性共同复测分别为7/24和8/24，第二轮标准起身也失败。没有新策略合格，最终按冻结选择规则保留R157的18/24；这不是新PPO成绩。没有改善、扩大或资格，full_task_completed、hardware_readiness和qualification均为false。

## 授权范围和新方法

R196是参考辅助的独立十腿总目标actor critic PPO，不是无参考自由动作策略，不叠加R157残差，不复用R195的RLS或MPC，也不扩大R181递归网络、噪声或训练轮数。固定reference仍在。控制1至528以reference加0.18乘tanh潜变量替换十腿旧反馈目标；控制0、四个头部目标和home529以后保持原接口。旧R157选择器、R122节点、IMU及右髋计算仅作为同状态旁路、首控制、头部和home来源，其起身十腿修正不加到新策略。

actor和critic各自为32、16隐藏单元网络，输入是当前native50、因果初始native50逐项tanh及固定控制进度，共101维。seed296仅初始化和离线实验RNG；actor末层初始零，不是全部参数零。无旧权重、教师拟合、case或seed标签、目录、root真值、qpos或qvel、未来读数、精确上下文查询或私有动力学输入。原framezaxis不是bodygravity。

原reference总修正界0.18rad、joint、5.24rad每秒乘0.02的slew、torque、50Hz控制、500Hz物理、碰撞mesh和flags、walker、reward和验收均不改。constructor40与reset40两批真实prepare分开保存，没有额外prepare、沉降、等待或中途root重置。没有连接、使能或部署硬件。

## 固定路径预算与退出

总上限162条保存的路径尝试，实际162，不是162独立起点。独立6完整smoke自然退出后，正式156包括同6启动一致性、25原旁路baseline、两轮各25完整采样训练和25确定性检查点共同复测、25最终候选。160条2279controls和45.58秒，另两条原物理无效分别303与304controls早退；共365247controls。全部失败保留，没有追加方向、幅值、参数、轮数或采样预算。

实际ROOT为/data/shijinsheng/open_duck，project为ROOT/projects/Open_Duck_Playground，SSH别名5090x8。启动器diagnostics.launch_getup_total_target_ppo_r196 --launch，后台同模块无参数；子diagnostics.train_getup_total_target_ppo_r196 --smoke，正式同模块无参数。OMP、OPENBLAS和MKL各1，CUDA空、JAXcpu，最多6CPU。

输出为ROOT/outputs下getup_total_target_ppo_r196_preflight_20261010、getup_total_target_ppo_launcher_r196_20261010、getup_total_target_ppo_r196_smoke_20261010及getup_total_target_ppo_r196_left_20261010；smoke和formal同名.log及ROOT/tmp/getup_total_target_ppo_r196_launcher_20261010.log保留。launcher记录natural_exit=true、smoke_exit_code=0、formal_exit_code=0、formal_not_run=false、smoke_gate_passed=true、source_hashes_unchanged=true。

## 回归和独立冒烟

独立离线20项回归2.862秒通过，exit0且new_dynamic=0；启动器初始20项2.786秒及smoke自然退出后的正式前20项2.859秒均通过。本轮没有训练修正版或回归失败。检查包含扰动组nominal_success为null时不误判标准失败，不修改R195保留的过严检查。

独立6均2279controls、45.58秒、原物理有效，共13674controls。原旁路标准和769000成功，773004失败有效，三条全部旧字段、initHash、原结果共有字段和peaks逐位R157。初始确定性新标准成功，11.10秒进入、严格尾34.68秒；actor均值零是相对reference零，不是R157总反馈零。其14合计修正峰值2.9567302083366442e-6rad来自原首控制、头部或home边界，不能称网络学习效果。初始采样769002成功，11.14秒进入、严格尾34.64秒；773004失败有效。不是6/6恢复，不构成共同提升。正式相同6全部新旧数组、原结果和RNG与独立smoke精确一致。

## 完整回合学习和共同复测

原逐步和终态奖励未改。完整Monte Carlo回报在实际回合终端截止，gamma为0.99的0.2次方，无跨reset bootstrap；完整home奖励进入恢复回报。控制0与home的actor似然mask为零，critic学习全部访问读数。潜变量高斯标准差固定1e-4，tanh后总目标再原限位；不把限位执行动作称为高斯。PPO比值截断0.8至1.2，学习率1e-6，最多4epoch、minibatch512、全局梯度范数1，整批近似KL大于0.02可提前停止，不增加epoch。

| 组别 | 原24扰动成功 | 标准成功 | 原物理无效 | 原return总和 |
| --- | ---: | --- | ---: | ---: |
| R157基线 | 18/24 | true | 0 | 1187.675118032883 |
| 第一轮采样训练 | 9/24 | false | 0 | 517.6520599956106 |
| 第一轮确定性复测 | 7/24 | true | 0 | 451.9468586452574 |
| 第二轮采样训练 | 5/24 | false | 2 | 249.05708629192117 |
| 第二轮确定性复测 | 8/24 | false | 0 | 452.950193993403 |
| 最终原旁路候选 | 18/24 | true | 0 | 1187.675118032883 |

两轮各4epoch，分别448与416次真实optimizer更新，actor和critic全部12参数块每轮都有非零变化。初始及两个检查点保存完整权重、Adam状态、RNG、训练批次、原step reward、完整回报、advantage、actor mask、统计、梯度范数、最后梯度及参数变化。没有保存可直接恢复在线环境的完整状态。

共同入选要求标准成功、全部物理有效、保留原所有成功且成功数严格大于18。两轮均不合格，selected_checkpoint=null、selected_R157_fallback=true。最终25条候选全部新旧轨迹数组与本轮25baseline相同，原结果共有字段和peaks精确R157。没有拼接成功并集或在线case oracle，没有以原return改善替代严格恢复验收。

## 两条原物理无效路径

第二轮训练769005保存303controls，initHash为172feeeaef32e321b88dc754bb33c8f0f1473995ba44a2a5a2aa7c18f59695bf，采样seed2963206；原峰值floor0.004164852893220163m、self0.04865534561987595m、joint0.010121218394570142rad、force3.23、slew5.2400000000000055rad每秒；14总修正峰值0.0022813584875467363rad，原return为-4.776136036253273。

第二轮训练773006保存304controls，initHash为da592b0c597f5f88206028383513b62762d1f4f17fd4c729386e64dfabce3e0c，采样seed2963215；原峰值floor0.0039603452426219005m、self0.0486179379156161m、joint0.010709671378986707rad、force3.23、slew5.2400000000000055rad每秒；14总修正峰值0.0046075462266774725rad，原return为-4.762488280601144。最终self读数较小不否定原500Hz峰值。

303或304及控制末端不是首次500Hz crossing；小动作和未达cap不证明安全，不用浅几何重标原失败。访问状态随后分叉不是同状态即时反事实或唯一失败根因。

## 独立只读核验与来源边界

diagnostics.archive_getup_total_target_ppo_terminal_r196 --base 92a8eba624c0c30748c80cf86eff331ac062ffca --verify已生成ROOT/tmp/getup_total_target_ppo_r196_terminal_verification_20261010.json，完整证明terminal_result_saved、natural_exit、new_dynamic=0。上次对话被用户打断，未观察到该调用的退出回执；不编造该session exit。后续同脚本归档入口无--verify会再次独立核验并要求与既有proof完全相同，再归档；实际归档exit另行等待与保存，不重新执行动态或重建--verify文件。

核验全部162条和365247controls，独立重算actor均值、创新、潜变量、log probability和采样RNG精确；从已保存postIMU/right exactdouble边界独立重算十腿替换、头部原样、之后joint和slew计划及真实执行递推。control0保留原接口，全部planned==applied。56原旁路路径全部旧数组、共有原结果、peaks和initHash逐位R157；qpos和qvel仅隔离全字段equality，不作控制或新推理。正式6与smoke及最终25候选与baseline全部新旧字段精确配对。

两轮完整learner批次的features、Monte Carlo returns、advantage、mask逐位重算，448和416次更新的统计、梯度范数、最后梯度、权重和checkpoint RNG精确；Adam learner序列化bytes精确。独立核验使用同JAX和optax算术，不冒称跨实现bitwise、动力学安全、反事实或训练有效。没有环境、MjData、forward、积分、contact、distance、实际force查询或原成败重标。

原before-request与完整IMU double未保存，whole_old_feedback_independently_recomputed=false；postIMU/right exactdouble是明确输入边界，不冒称重建全部旧反馈。四个test源副本经hash验证，但preflight和两轮test的启动路径映射manifest未保存，test_startup_mapping_manifest_not_saved=true。已有副本和缺口原样保留，不事后回填。main、worker和launcher有显式四源码及启动已导入Python副本，不是第三方native binary快照。

完整XMLinclude和assets、frozen输入和源码前后hash不变。compiledMJB为53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3，原physics为4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae。kernel SHA为62cef884750014c7e33835351e800c03e90bb14b74fe793d9e7f85d34e8c70fd，main为30538eb01571f117979ad3d3292e5990511c39a9537cf68cfbf6bf6aa10a16bb，test为b4755e5f017c6d8002d5fb70fee594e21dc0544ce86d7c16a32b072365ed1c2c，launcher为059cded23a1d9532bebe33c14760de7ef23bff52101ee7ca3ea93b48b3b67a7c。

## 归档发布和后续门槛

本记录准备时发布库main为92a8eba624c0c30748c80cf86eff331ac062ffca且clean，仅受保护live_server2664890，没有训练或publisher运行。可用19963994112bytes是该轮快照。四个新live目录合计488795522bytes，包含preflight；main自动storage guard不包含preflight，独立归档预算明确把它计入。live、archive及Git增量合计必须小于2GiB，开始至少12GiB、持续保留10GiB，不清理他人或未确认数据。

新归档只追加四terminal_snapshot、全部失败与原peaks、完整检查点和RNG、learner批次与核验proof、源码、计划、本终态和唯一publisher，不复制旧dynamic或覆写旧证据。只有新归档真实exit0、唯一新commit clean、唯一publisher正常FF真实exit0、成功新receipt和原helper无--push独立核对同SHA，再GitHub明确commit全文一致后，才称发布完成。本段为归档前条件历史，不预称发布成功；实际回执另行保存。唯一publisher名publish_getup_total_target_ppo_terminal_r196_20261010.py，新receipt名getup_publication_r196_total_target_ppo_terminal_20261010_result.json；不复用旧receipt、竞争push、强推、下载bundle或导出key。

本轮没有提升，统一最佳仍R15718/24。不重跑R196、不追加两轮或噪声网络预算。下一步必须真正不同有限设计，先声明结构区别、因果输入、离线用途和动态存储预算，原回归及独立完整smoke自然退出后才formal，不绕过失败门禁。原24至少22且优于18/allvalid、冻结316至少36/40、未见320两组20各18以及更大扰动、传感执行延迟、right、prone、supine全部门槛保留。320仍未读取执行，全部仿真验收未过，不标任务完成或自动部署硬件。
