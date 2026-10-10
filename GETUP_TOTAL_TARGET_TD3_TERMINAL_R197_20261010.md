# R197 独立双Q经验回放起身训练终态

## 结论

R197 两轮训练、独立冒烟与正式共同复测已经闭合，启动器记录 natural_exit=true、smoke_exit_code=0、formal_exit_code=0、source_hashes_unchanged=true。两轮确定性成绩9/24与6/24，均不合格，无统一改善。最终18/24是冻结R157回退，不是新TD3成绩；统一best仍R15718/24，fulltask/hardware/qualification=false。

实验目录沿用20261010标识；闭合核验续至上海时间2026年10月11日。此文准备时独立只读核验已自然exit0，归档和上传尚须执行并取得真实退出、收据及独立远端/明确GitHub提交核对。本文件中的准备阶段条件保持历史，不以后续完成事实覆盖。

## 实际执行与预算

唯一启动器 diagnostics.launch_getup_total_target_td3_r197 --launch 请求真实exit0，后台同无参；独立子 train_getup_total_target_td3_r197 --smoke 自然退出后，同无参正式。原环境/.venv、OMP/OPENBLAS/MKL各1、CUDA空、JAXcpu，实际affinity前六核。旧launcher1685855、smoke1686147、formal1690035均只定位；闭合后只剩保护中的live_server2664890，不操作旧PID或重启服务。

ROOT=/data/shijinsheng/open_duck，项目projects/Open_Duck_Playground，SSH5090x8。新输出：

- getup_total_target_td3_r197_preflight_20261010；
- getup_total_target_td3_launcher_r197_20261010；
- getup_total_target_td3_r197_smoke_20261010；
- getup_total_target_td3_r197_left_20261010。

离线22回归8.209s真实本机session44373自然exit0；启动器初始22回归8.402s、正式前22回归8.168s全过。测试不构造动态回合。真实smoke和formal日志、源与映射、原准备帧、全部结果另存。

实际162次闭合路径尝试=独立6+正式156，非独立起点数。正式为6startup+25baseline+50训练+50确定性复测+25最终回退。161条原2279controls/45.58s，一条原invalid304controls，合计367223controls。原invalid早退不补造走满。

固定预算162、两轮、最多6CPU不追加。本次独立proof统计live521645671bytes，包含preflight/launcher/smoke/formal；3倍live保守预留小于2GiB。新storage_gate确实包含preflight，与R196内部自动检查未涵盖preflight的历史缺口不同。实际归档+Git增量仍须归档退出时统计，启动free至少12GiB、reserve10GiB；历史free不是未来保证，不清理他人/未确认data。

## 新方法及与旧结构的区别

新seed297的离策略双动作价值网络、目标网络、经验回放TD学习与延迟确定性actor梯度。受[TD3原论文](https://proceedings.mlr.press/v80/fujimoto18a.html)启发，但恢复/home边界处理和极小固定预算是本实验有限变体，不冒称原论文训练规模或保证。不是R196 MC/PPO概率比、R181整回合score更新，不复用任何旧训练权重、Adam、轨迹或批次，不增加旧方法预算。

actor输入101=当前actualnative50与因果初始native50各tanh+固定进度，网络101→32→16→10，输出tanh归一化总请求a；last初始零，其余新随机，全部学习。Q1/Q2各自111→32→16→1、新独立随机权重，输入101加本次请求a。没有在线case/seed/path/label/root/qposqvel/teacher/未来/lookup/privatephysics输入。原framezaxis不是bodygravity。

controls1..528十腿目标reference+.18*a替换旧腿反馈，不叠加R157残差，不再tanh(a)。采样a=clip(actor(x)+1e-4*normal,±1)，不称限位后的执行动作高斯。control0原目标，四head原样，home529+原feedback；R157/R122/IMU/right只同状态旁路及首/头/home。原joint/5.24rad/s*.02slew/torque/50与500Hz/mesh/flags/physics/reward/验收不变。

只新R197训练回合进入回放，包括全部原失败，不混baseline或旧dynamic。gamma=.99**.2，恢复下一步仍active时TD目标用min两个targetQ；k528进入home用本次完整原记录home折扣尾回报；invalid/终端截止、不跨resetbootstrap。离线下一观测只在真实到达后供学习，不是在线未来输入。价值估计不是物理效应预测或安全证明。

每轮64次Q更新、32次actor更新，batch512有放回。Qlr1e-4、actorlr1e-6，各Adam/globalnorm1，delay2、tau.005，目标噪声1e-4/clip±2e-4/动作±1；目标stopgradient。initial+两完整checkpoint含在线/目标网络、两Adam/count/RNG；没有在线env完整恢复状态。第一轮13200新恢复转移，第二轮13200、累计26400；这不是新的动态次数或独立起点。

## 共同成绩与选择

| 阶段 | 原24成功 | 标准成功 | 原invalid |
|---|---:|---|---:|
| 冻结baseline | 18 | true | 0 |
| 第一轮采样训练 | 8 | false | 0 |
| 第一轮确定性复测 | 9 | false | 1 |
| 第二轮采样训练 | 9 | false | 0 |
| 第二轮确定性复测 | 6 | false | 0 |
| 最终R157回退 | 18 | true | 0 |

两轮actor/Q全部18块参数有非零变化，但这不等改善。共同选择须Std、allvalid、保留baseline全部成功且>18；按success/min原return/sum原return排序。两轮均ineligible，selected_checkpoint=null、selected_R157_fallback=true。最终25全部新旧数组与baseline精确相同；不做成功并集或在线caseoracle。22门禁未过，不扩大316、不读/执行320。

独立6均完整2279controls/物理有效，共13674controls，真实进程自然exit0后才formal。baseline标准/769000成功、773004失败有效；所有旧字段/peaks/initHash逐位R157。初始新确定性标准成功11.10s/strict34.68s，actormean0相对reference，不等旧feedback0；全14combined峰值2.9567302083366442e-6来自首/头/home原项，不称网络已学。初始采样769002成功11.12/34.66、773004失败有效；不是6/6起身或统一学后成绩。正式同6全部新旧数组/result/RNG与独立6精确。

## 原无效与解释边界

唯一新invalid是第一轮确定性773005，原保存304controls。原peaks：floor.0046660449273014095m、self.04870735503909062m、joint.012367144384208772rad、force3.23、slew5.2400000000000055；14combined修正最大.0004098126522655843rad。原结果、初始hash、RNG和完整失败保存，不重标。

304控制记录/控制端点不是首次500Hz crossing。小动作、未达cap或较小终态self不证明安全，不用浅geometry推翻原峰值，不将不同visited状态分叉推为同状态即时反事实或唯一根因。此失败不证明所有TD3方法无效，也不授权追加本方法预算。

## 独立只读重算

新archive_getup_total_target_td3_terminal_r197 --base 0c56db66bc246ab8649b9f130475400f3390ec34 --verify，本机真实session80721自然exit0，保存独立proof。无新动态、env/MjData/forward/积分/contact/distance/实际force查询或成败物理重标，不调用R195 init_worker追加旧capture。

全部162/367223控制的actor均值、采样创新/归一化clip请求、RNG由真实current/initial传感器和初始/各checkpoint重算精确；原selector/right由actual与冻结数据重算并核对。postIMU/right exactdouble边界后十腿覆盖/head原样，以及第一步之后joint/slew/base/adjusted/递推prev/planned==applied精确；第一applied逐位R157。56baseline路径全部旧字段/共有原result/peaks/initHash逐位R157，qpos/qvel仅隔离equality。全部原stepreward sum与原return_sum既有1e-10累积界检查。

两轮独立重建新转移、home完整backup、terminal截止、累计replay与manifest；重生128批索引与目标噪声；重算128次Q/64次actor损失、norm、lastgrad、在线/目标权重、两Adam/count序列化msgpack字节和RNG全部精确。未调用新kernel.Learner来决定核验结果，使用相同JAX/optax，不能冒称跨实现bitwise、学习有效或安全。

核验前后四输出完整文件inventory相同，全部冻结输入/source/modelassets前后hash不变。三次test先导入learningruntime再保存四显式源与全部已导入Python及mappingmanifest，已全部核对；main/worker/launcher同样保存Python启动副本。非nativebinarysnapshot。R196 test启动mappingmanifest缺口不回填；旧before-request/完整IMUdouble未存，不冒称独立重建全部旧feedback。

完整编译MJB53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3、原physics4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae不变，完整XML/assets/frozen/source保持。

## 固定源码与归档发布条件

本机/服务器/启动显式副本已hash核对：

- kernel f7dd0e6ab1f5121eb25a5f03983d2e056350f100d684b9d29f9a7367633c6b0d；
- main ce69903501649adeb80424a7d04b7f3a643c34e1264d41fc2aa4fe2f15aeb1a6；
- test 581e742d0605d59a5ca08b34443afec9a95a23cd1909a0c74cbac34cb9c8b5ac；
- launcher eab0e6c0675e5e55eeb136841ff597343cf540ac38fb41e1ff5cc3413751bf71；
- archive 9352d5736b6904e7ac0fdd6599818d6b37b794507bf041284a01b9f6c75a834e；
- publisher eea1d619df2cd02505d664cdccecabb18e526734c85993ff1010e48c6eebc7f0。

新归档须再次同程序只读重算并与proof精确相同，追加四terminal_snapshot、全部失败/学习器/source/RNG/proof/新docs/scripts，不复制旧dynamic或覆写旧证据。新干净提交父0c56；真实archiveexit0后才能唯一publisher --expected完整SHA --push正常FF，等真实exit0和新receipt再原helper无--push独立核对remoteMain/localHEADclean，最后GitHub明确commit完整本doc一致。未满足之前不称发布完成，不竞争push/强推/导出key/下载bigbundle/复用旧receipt。

## 续跑边界

本轮固定两轮已用尽且不合格，不追加TD3网络/噪声/参数/回放/批次/轮数/window；R196/R195及旧方法也不扩预算或重复动态/审计。下一步须真正不同有限方法，先声明结构差异、因果输入、offline用途和固定动态/存储预算，通过原接口/边界/init/回归及独立完整smoke自然exit后才formal。没有选定或启动R198，不能套用旧“下一步R197/正在训练”的状态。

原24>=22且>18/allvalid→316>=36/40→未见320两组20各>=18→更大扰动/传感执行延迟/right/prone/supine全部既有门槛不变。320仍未读执行；全部仿真验收未过，不标fulltaskcomplete，不连接/使能/部署硬件。当前自动化ACTIVE30分钟仅当时读取快照，保留未来实际设置，不依据历史ACTIVE恢复人工暂停；只实质进展/异常/需用户时通知。

