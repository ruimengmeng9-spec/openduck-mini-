# R192 速度惯性偏置与原目标限位独立审计终态

2026年10月10日。独立只读6条及正式62条既有 R191 轨迹审计均已保存并自然退出，没有新动态回放、训练、控制器拟合、扩大或资格执行。两轮11项新回归分别1.057秒、0.998秒全通过，无失败或修正版。R191候选10/24、标准成功、physical invalid 0、gate false，冻结R157基线18/24保持；统一最佳仍18/24。

## 实际命令与范围

ROOT=/data/shijinsheng/open_duck；project=ROOT/projects/Open_Duck_Playground；SSH别名5090x8。实际启动为现有 .venv/bin/python -u -m diagnostics.launch_getup_velocity_bias_audit_r192 --launch，后台同模块无参数。启动器先新 diagnostics.test_getup_velocity_bias_audit_r192，随后同python -u -m diagnostics.audit_getup_velocity_bias_terminal_r192 --smoke独立退出，再11项新回归和同审计模块无参数正式。OMP_NUM_THREADS、OPENBLAS_NUM_THREADS、MKL_NUM_THREADS均1，CUDA_VISIBLE_DEVICES空，JAX_PLATFORMS=cpu，MuJoCo3.12.0；没有重跑R190离线20项或R191动态。

输出为ROOT/outputs/getup_velocity_bias_terminal_audit_r192_smoke_20261010、getup_velocity_bias_terminal_audit_r192_20261010及同名log；独立启动与两轮回归证据在getup_velocity_bias_audit_launcher_r192_20261010，启动日志ROOT/tmp/getup_velocity_bias_audit_r192_launcher_20261010.log。启动器 result.json 的 natural_exit=true、exit_code=0、new_dynamic_trajectories=0、source_hashes_unchanged=true。

独立六条选自R191正式：zero标准、bias标准、baseline769000、candidate773004救回、769000退化、769002失败有效。三扰动均实际访问非零偏置。正式审计R191独立6与正式56全部62条，141298控制记录=62×2279，原轨迹均45.58秒完整且有效；不是新轨迹或62独立起点。失败不排除。25终态paired initHash、baseline全部原字段与peaks逐位R157一致；独立6行、25配对和所有对应信号数组与正式逐位相同。保存正式28信号（25candidate加zero标准、bias标准、baseline769000）及独立6信号。

## 独立重建的决定

只从actual native34 gyro0:3、14 home-relative位置6:20、速度20:34除.05、因果initial及固定同相位nominal独立重建。up3:6为framezaxis，本偏置不用；没有actual root、qpos/qvel、case、seed、目录、label、teacher、未来actual或lookup进入偏置决定。

独立类重新映射gyro site、同weld刚性组、site rotation与angular Jacobian、自由根角速度坐标、原servo unit gear、fixed gain/affine position/no internal dynamics、kp13.37Nm/rad和十腿索引。canonical root位置0、单位四元数、根线速度0。仅mj_kinematics、mj_comPos、mj_jacSite、mj_comVel、mj_rne(flg_acc=0)，同q原gravity下移动速度与零速度RNE差给b，独立重算e=(b-bnom)-(bi-bnom0)、raw腿=+e/kp，head新增0。独立代码不调用旧kernel的measure或request决定函数；保存bias/error/raw/moving/static/private reconstructed velocity仅比较。

新回归与同C API实现逐位核对20个合成测量、请求、实际529名义零、control0因果零、gyro/weld/native单位、原servo正号、head分支、未用up、merge/限位/对象、禁止private动态、完整MJB不变、白名单、smoke/固定源及非法输入。没有重新声称跨C/Python实现bitwise；R190独立Python递推原atol1e-14/rtol3e-12算术证据保持历史。

冻结R157真实初始native50 selector和原右髋feedback从actual preparation/native重新核对逐位。original_double_pre_slew_target_rad已postIMU/right，是明确双精度输入边界；其之前的原IMU double与request未保存，不冒称重建全部原IMU。从此边界独立重算reference±.18、十腿merge、head原样、joint、5.24rad/s×.02slew、base、adjusted、pre/post直接差、planned、actual applied和递推previous；全部62逐位一致。previous的native float32编码也精确核对。control0保留原对象；standard raw/error精确0；home529之后新增所有信号0，原feedback照旧。standard b/moving/static可非零。

无environment、forward、inverse、积分、contact、distance、passive、actual force query或成功/物理标签重标；没有写原模型、physics力、reward或验收。qpos/qvel仅隔离全字段equality解码，不进入scalar/planning；hash读取完整压缩bytes不等root解码供反馈。

## 跨案例结果和限制

救回773004 raw=0.006837605094433795rad、pre/post=0.006837605094433791rad、total=0.13306071774328443rad、e最大0.09141878011257984Nm，原11.10秒进入/34.68秒strict尾。退化769000 raw/pre/post=0.008489248602414817rad、total=0.14909234467863963rad、e最大0.11350125381428611Nm，原失败有效。769002 raw=0.003634974940968341rad、pre/post=0.0036349749409683415rad、total=0.13399598426728154rad、e最大0.05554625394611734Nm，原失败有效。全部24扰动post差首次control1，raw峰值0.0023738400696582023至0.009215486300679521rad，合计峰值0.18。

R191原10成功、救回3和退化11以及所有labels/peaks保持原记录。独立代数/限位一致不等20ms动力学响应、根加速度/接触/耗散、安全、恢复改善、即时反事实或唯一根因。head与腿在规定根运动及qacc0时为不同分支，head速度不影响腿b；不冒称整机head协调腿。偏置不含gravity、Mqacc、constraints或actual root线速度。小raw、cap或控制端点不能重标原500Hz峰值，不追加本偏置幅值/方向/window/model参数预算。

## 来源完整性与存储

main SHA256 2a691e8d5d7d1e0af17feaa9b90b2be8ffe7f130733b358646067322e703cbcd；test 047273d34efbb691138941f7695b3d3f1f4a5c94d6eed498736accbcd2e16960；launcher 427fd3415241586287ef5681fb452dd2f97d2576e17a17c8a3dd2d2d7c656e10。本机、服务器和显式main/已导入全部Python执行副本一致；启动器、两轮test、独立/正式audit均保存startup sources，不是第三方native binary快照。

固定R190kernel SHA44d1f4881ac612e66c2a99fa7bceab16ce0fd4657f3d6bb2089077db3f2d07fa、R191probe SHA8871d666a5ea6a60a472db72f0978404f1c3cc3c2797cbd3d9292b1163bb7fbe不改。source/evidence/frozen/reference/stand/全部XML include及file assets hash前后不变，实际完整compiled MJB SHA53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3前后不变。原R191每条oldphysics SHA4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae记录保留；本审计不重新积分physics。R190失败原全部bytes保持，R188及更早缺口不回填、不改旧startup/partial/terminal/source/docs。

新独立19678777bytes、正式37397897bytes、启动器32265751bytes，活证据合计89342425bytes（launcher收据写入前记89342191bytes）。预算0.25GiB，至少10GiB保留；新归档只复制本审计，不重复制R191动态。归档由新archive_getup_velocity_bias_audit_r192.py验证来源/配对/信号后保存两terminal_snapshot与launcher terminal。只追加新scripts/doc/evidence，不覆盖旧失败、固定源、归档、publisher或收据。

## 归档发布条件与后续

归档基线为已发布clean f31fe555793beb8ee54d124f91d994a98c58b847。本文生成时新归档与上传尚未执行；发布完成仅以后续新commit、唯一publisher成功exit、成功receipt、原helper无--push独立只读及GitHub明确commit读回为准，不能由本段条件宣称上传成功，也不覆写本文历史条件。发布遵守AGENTS.md与OPEN_DUCK_SERVER_PUBLISH_20261009.md，不导出key、不强推、不竞争上传。

下一轮先实时查R193或更高actual source/output、完整命令、最新main/clean、资源。健康任务等待，不重启/发信号/改预算，保护live_server2664890。只有无更高任务且新发布完成，才依据完整跨案例证据提出真正不同有限全过程因果反馈或多真实倒地共同完整复测稳健教师；先说明区别、因果输入、offline用途及动态/存储预算，教师与root truth只offline监督，不online oracle/context lookup。不要重复本审计或R190/R191，不扩速度偏置/momentum/pinv/旧PG/phase/projection/BC/静态插值或旧禁结构。

18/24不跳扩大或资格；320仍未读执行。原22/24且优于18全部有效→316至少36/40且优于zero/R102全部有效→未见320两不重叠20组各18/20、12秒内进入、strict连续30秒、逐500Hz有效，以及后续更大独立扰动/传感执行延迟/右侧prone supine各20未见18/20门槛不变。ACTIVE每30分钟，健康和相同连接不可达无变化静默。full_task_completed=false、hardware_readiness=false、qualification=false；只有全部既有仿真门槛通过并保存证据、列硬件未过项才结束，不自动连接、使能或部署真机。

