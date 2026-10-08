# R168完整终态与R169实际专家输出审计（2026-10-08）

## 结论

R168四代有界训练及最终完整开发已经结束并自然退出。最优仍48参数全零，统一候选与冻结R157基线均18/24，标准成功、物理无效0，development gate=false。没有新提升、扩大或未见资格。R169只读重算625训练尝试和25终态配对，已结束；不是新动态回放、训练或在线教师。全部失败和旧失败来源保留，任务继续ACTIVE，不称仿真验收完成或真机可用。

## 实际执行与闭合数据

项目 `/data/shijinsheng/open_duck/projects/Open_Duck_Playground`，现有 `.venv`。

R168实际完整命令：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -u -m diagnostics.train_getup_expert_mixture_r168
```

输出 `/data/shijinsheng/open_duck/outputs/getup_expert_mixture_r168_left_20261008`，同名.log。seed268、4代8提案/6CPU、48系数±1，CEM mean0/std.02、elite2、.6/.4更新、std.002至.05，zero/best保留并缓存重复。25实际程序（24非零）、625完整路径训练尝试，4代参数/搜索分布/报告/history/learner RNG/checkpoints及training_closed.json均闭合。每次尝试使用原2279步45.58秒、12秒进入/连续严格站稳30秒/500Hz逐子步物理验收；物理无效可以提前终止，不能称625条都走满45.58秒。

最终candidate/baseline各25完整开发，另有6正式启动一致性，共681正式动态尝试。旧独立6完整smoke加上后为687，不是687独立起点。12回归、独立6、正式6逐位一致性已由原startup保存。第一版随机代数凸界断言0.18000000000000002严格比.18失败，无动态回放；原失败三源码/日志保留，仅测试加1e-14舍入容差，执行kernel、目标限制和原验收未改。

## 控制器与因果边界

冻结R157实际初始native50选择器、R122snapshot程序/节点及原IMU反馈。同一当前真实状态计算四套冻结R133右髋反馈。当前/初始gyro、up、右髋位置速度减同相位名义，尺度gyro1/up.05/pos.05rad/nativevel除.05恢复rad/s；当前误差再减因果初始误差，phi=tanh(delta)。W4x12，gate_i=max(0,tanh(W_i dot phi))，mixed=(base_feedback+sum(gate_i*expert_feedback_i))/(1+sum(gate_i))。混合tanh之后的反馈输出，不平均gains；不是静态选择器再拟合、旧动态gain或直接残差。

控制0 phi/gate精确零、直接原base反馈，第一目标R157逐位；实际标准scalar各路精确零，零参数直接原反馈。529步后home原反馈。只替换原三右髋local反馈，其余11关节和IMU保持。14joint原新总参考±.18rad，原joint/torque/slew/home/碰撞mesh及标志/500Hz物理/奖励/验收不变。无case/seed/目录/标签/root真值/未来子步/精确上下文或近邻lookup控制输入，无新增等待/沉降/prepare积分或中途root重置。

四专家在其他访问状态是否能恢复未知，凸界不证明接触安全、能量耗散或成功。原PROBE冒烟救回769002不能推广统一成绩，不补做同PROBE。

## R169只读核验

新源码 `diagnostics.audit_getup_expert_mixture_terminal_r169`。独立smoke使用 `--smoke`，先6已有轨迹scalar审计与25终态配对，自然退出后正式运行：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m diagnostics.audit_getup_expert_mixture_terminal_r169
```

正式输出 `/data/shijinsheng/open_duck/outputs/getup_expert_mixture_terminal_audit_r169_20261008`，smoke为同名加`_smoke`目录，均独立新目录。正式全部25程序/625尝试实际phi/gates/5路凸权重/4专家输出/base/混合/三髋反馈差逐位重算相同，控制0及标准/home新增零。25最终candidate/baseline所有保存字段、初始hash、原物理peaks逐位相同，baseline原字段逐位复现R157。sourcehash前后不变，正式对应smoke数组/统计逐位一致。

scalar仅读观测/执行目标/local反馈/专家/gates/weights/phi/原差值白名单。qpos/qvel仅在单独全部原字段一致性核验读取，不作反馈输入；initial audit未用root监督控制。没有环境/MjData/forward/积分或力推断，不重标成功、物理有效或峰值。

保存全部625标签与统计，最高非零及最高全有效各25完整配对信号数组。原始三右髋同状态反馈差与14路已执行目标差分别保存，不能混作同状态即时增益、动态反事实或唯一原因。

## 非零候选的完整结果

24非零仅8套对全部训练起点物理有效。最高非零g4/c06完整12/24、物理无效1，救回769004/773004，却退化769000/769005/773006/773008/773009/773010/773012/773014。773006保存303控制步、self峰值.04866400915770342m，原其他物理项未超；这是保存控制终点，不是首次越限500Hz子步。maxgate.0079884、专家总权重.0123501、同状态三髋反馈差峰值.0000556762rad、原新总最大.0916873rad，不支持该例简单达到±.18上限解释，也不证明小混合安全。

最高全有效g3/c06完整9/24、物理无效0，救回769004/773005/773015，却退化769000/769001/769003/769005/769006/773000/773002/773003/773009/773010/773011/773013。24扰动maxgate范围.0211562至.0445412、专家总权重.0266150至.0549547。前50控制步原始三髋同状态反馈差峰值.00000612557至.000447379rad，14路已执行目标差峰值.00000617602至.00737156rad；维度与语义不同，不作相除传递增益或恢复反事实。

该程序三救回首次目标差标记分别控制1/2/2，传感器差控制2/3/3，1e-8只是诊断标记。未证明唯一失败原因或所有状态反馈无用。拒绝将已用标签/离线专家并集作在线oracle，也不追加同48矩阵预算或补固定PROBE。

## 追加保存与后续

新归档脚本 `archive_getup_expert_mixture_terminal_r168_r169` 从已核对GitHub/server干净基线a74649ad5d08bd0bdf4b314d928aa7d44fbac28a追加唯一terminal_snapshot。R168保存完整4代/25实际程序/681正式尝试及全部失败、学习器RNG、冻结执行源/模型/物理hash和回归日志；R169正式及独立smoke保存只读数组/标签/原峰值/终态配对/hash。快照terminal_result_saved=true但full_task_completed=false、hardware_readiness=false、资格未执行。

旧R167268fc、R168 startup81c0ea、g1a74649已在本轮正常快进补推且独立核对。旧startup/g1只记录当时状态不冒称终态；固定目录/脚本/bundle、旧失败来源不覆盖。新唯一bundle大小/SHA256由归档输出，传输完成后校验依赖、fetch、正常快进push并独立只读核对才算上传成功；本文件不预先声称传输完成。

下一轮先核对R170或更高实际任务/完整命令防重复。依据完整跨案例耦合提出与旧预算明确不同的新有限方法，优先全过程统一因果反馈或多个不同真实倒地共同完整复测的稳健教师，不继续方向阈值、静态成功集合拟合/精确插值、旧gain/直接残差/BC聚合/几何时间窗。先因果/边界/实际scalar名义零/初始化与完整原轨迹逐位一致回归和独立smoke再训练，方案未验证不称改善。

标准保留、原开发>=22/24、优于实际18基线且全有效，再新目录冻结扩大316>=36/40并优于fixedzero/R102全有效，才资格。3200000至3200039未读取执行，316/318已开发；冻结candidate/zero新40起点hash配对，两20组各18/20、12秒进入/严格30秒/原500Hz有效后还需更大扰动及延迟、右/俯/仰各20未见18/20。没有连接、使能或部署真机，不改原行走控制器、不清理他人或未确认数据。
