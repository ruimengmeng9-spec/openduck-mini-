# R193 全恢复段共同教师终态

2026年10月10日，R193 的独立完整 smoke、五个共同教师程序和最终配对复测均已闭合并自然退出。四个非零程序没有同时保留原成功并提高统一成绩，最终选择零程序。候选和冻结 R157 基线均为 18/24，标准成功、物理无效 0；开发门禁未通过，没有扩大开发、未见资格或硬件部署。

## 完整流程和结果

初始 21 项新接口回归 1.347 秒通过；独立 6 条 smoke 自然 exit 0 后，第二轮 21 项回归 1.216 秒通过。正式 6 条启动一致性、5×25 条共同教师训练、25 条基线和 25 条候选，共 181 条正式路径；加独立 6 条共 187 条，不是 187 个独立起点。

183 条走满原 2279 控制步、45.58 秒；4 条因原物理有效性检查早退，保存 45/304/304/304 控制步。全部共 418014 条控制记录。原 12 秒进入、连续严格 30 秒、逐 500Hz 有效门槛不变，失败不排除、不改判。

| 共同程序 | 扰动成功 | 标准成功 | 物理无效 | 入选 |
| --- | --- | --- | --- | --- |
| 0 零程序 | 18/24 | 是 | 0 | 零回退 |
| 1 第一方向 | 11/24 | 是 | 1 | 否 |
| 2 第一方向反号 | 8/24 | 是 | 2 | 否 |
| 3 第二方向 | 8/24 | 是 | 1 | 否 |
| 4 第二方向反号 | 11/24 | 是 | 0 | 否 |

共同入选要求标准保留、全部物理有效、保留基线所有原成功且成功数严格提升；合格按成功数、最差原 return、总原 return 排序。没有合格非零程序，不能用各程序成功并集替代单一可执行候选。program 4 虽全有效，仍退化 9 个原成功；共同成果不成立。

各程序原 preserved/rescued/regressed 完整列表、每例原标签/峰值/报告、五个 closed_program、全部 proposals/selected_teacher、完整 proposal 与 terminal RNG 保存在终态证据。program 1 救回 769002/769004/773015；program 2 救回 773001；program 3 救回 769004/773005；program 4 救回 769002/769004。这些局部救回不能冒称统一提升。

## 原无效记录

| 程序和案例 | 保存控制数 | 原 self 峰值 m | raw 最大 rad | 同状态 post 最大 rad | 合计修正最大 rad |
| --- | --- | --- | --- | --- | --- |
| 1 773009 | 45 | 0.030023331742615438 | 0.000004240739854744184 | 0.000004240739854749487 | 0.03079548802773502 |
| 2 769002 | 304 | 0.0486417177943165 | 0.00003122813561817772 | 0.00003122813561817761 | 0.10355875100205791 |
| 2 773012 | 304 | 0.04874911310676949 | 0.000024373618437766017 | 0.00002437361843776742 | 0.10183239971154973 |
| 3 769002 | 304 | 0.04863884847461809 | 0.000029576244082290267 | 0.00002957624408228554 | 0.14780422342023036 |

控制末端和记录数不是首次 500Hz crossing。小动作、未达 cap 或控制端点不证明安全，不用浅几何重标原峰值；随后 visited state 分叉不是同状态即时反事实或唯一根因。

## 固定共同教师和因果边界

新程序为十腿全恢复段四个 global sin(pi*n*k/528) 波形的 40 个共同系数；不是旧三状态节点/短尾教师、旧 PG/递归读出、几何/phase/momentum/bias 的扩大预算。seed 293 一次生成两个 normal std 1e-5 rad 方向及反号，合计四个非零和零；系数限 1e-4 rad，没有追加模式、方向、幅值、窗口或自适应轮数。

在线仅 current native34、固定同相位 nominal 和因果 initial 差；gyro/up/q/nativev 的固定尺度为 1/.05/.05/.05，a=tanh(RMS(delta))，rawlegs=a*C*4harmonics、head 新增 0。up 是原 framezaxis，不冒称 bodygravity。无 actualroot/qpos/qvel/case/seed/path/label/teacher/未来/lookup 输入，无 private MjData、geometry、forward、RNE、积分、contact 或 force 查询。所有教师都从标准和原 24 真实完整倒地共同复测，无在线 case oracle，没有拟合教师 MSE 或新增 BC。

冻结原 R157 真实初始 native50 selector、R122 snapshot nodes、原 IMU 和右髋反馈。post IMU/right 的 exact double target 是明确输入边界；未单独保存此前 request/IMU double，不能冒称重建全部原 IMU。十腿 merge/head 原样、合计 14 reference±.18，再原 joint 和 5.24 rad/s×.02 slew/torque。zero 原对象、control0 原 target、标准新增精确零，home529 后新增信号零而原 feedback 保留；原 constructor40 与 case reset40 两批 prepare/time 分开保存，没有新增积分、等待、沉降或中途 root reset。

## 独立终态核验

新归档前检查从完整闭合记录独立重算五个报告、共同 retention/selection、seed293 proposals 和 RNG、每个 closed_program 的完整历史；所有选择一致。独立 6 与正式同 6 的全部新旧数组、完整结果、initHash 和 peaks 精确相同；最终 25 对 candidate/baseline 全数组及完整结果相同。87 条零程序或标准路径的全部旧轨迹字段和共有原结果字段/peaks 与 R157 精确相同。qpos/qvel 仅隔离全字段 equality，不进入新增控制或规划。

187 条全部 planned==actual applied 精确；原 source、frozen、全 XML include/file assets、MJB 及 physics 记录逐条核对不变。14 份 main/worker/两轮 test/实际 launcher 的 startup source manifest 与原源码及副本 hash 一致，包含 explicit main 和所有已导入 Python 源码，不是 native binary snapshot。全部轨迹压缩 bytes 与每个 result 的 hash 被记录。此核验是终态配对、保存信号和离线选择核验，不是新增完整 scalar/merge/planning 决定审计，不构造新动态，不证明恢复安全、跨实现 bitwise、反事实或唯一根因。所有旧 partial/source 来源缺口保留、不回填。

首次归档前只读核验错误地要求 R157 专属报告注解 baseline 等同名存在，KeyError 后 exit1，尚未归档、未改 repo、newdynamic0。原 archive_r193 源码和 archive_verification_failure_01.json 保留；新 archive_r193b 仅区分明确列出的八类旧报告注解与原共有结果字段，仍核对全部旧轨迹字段/原 peaks/共有结果，不改训练 kernel、控制规则、物理或验收。新核验已 exit0，没有训练或 launcher 修正版，也未追加回放。

## 执行和来源

ROOT=/data/shijinsheng/open_duck；项目 ROOT/projects/Open_Duck_Playground，SSH 5090x8。原 launcher 为 .venv/bin/python -u -m diagnostics.launch_getup_shared_horizon_teacher_r193 --launch，后台同无参数；独立 child 为 diagnostics.train_getup_shared_horizon_teacher_r193 --smoke，正式同无参数。OMP/OPENBLAS/MKL 各1、CUDA 空、JAXcpu、6CPU、MuJoCo3.12.0。launcher 收据 natural_exit=true、smoke/formal exit0、source_hashes_unchanged=true。

固定 kernel SHA256 3edcc26f9fb4d36b4d645c7d5834289e77f9f66e8cdf681f28f5c37990f2afde；main 8f26bc2e103f5489380c836fb02fc1e23e26e6485df95cd6830b7cc15ba7678b；test e92a30784bf1b9ee12fff5b80364995030214dff6f923b7d3849be4432bb9bdb；launcher abd8f4195b5ca58de4631ee3631250af57cb4219dddd5bafdd218bda1aab9ae3。两端及实际启动副本一致。compiled MJB 53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3、旧 physics 4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae 不变。

正式 live 405549383 bytes、独立 53491816、launcher 31000100，合计 490041299 bytes。归档前预留 3×live<2GiB；新终态复制和 Git 增量总预算2GiB、保留10GiB，不清理他人或未确认数据。只新 R193 三个 terminal_snapshot 和核验收据/原失败/全源/新 plan/terminal/archive/publisher，未复制旧 dynamic 或覆盖任何旧证据。

## 归档发布和后续门槛

新归档只在自然退出、当前 clean base cc3687ee8c877b348ac04f146d9733bec44f8e9a、无 active diagnostics/publisher/gitpush 后执行；以模块入口 archive_getup_shared_horizon_teacher_terminal_r193b --base 明确 SHA，保存原 archive_r193 失败源码和修正源码。发布须等唯一新 publisher 成功 exit0 后，再独立原只读 helper 核对同一明确 commit 与 clean repo，并从 GitHub 明确 commit 读取本文。准备此记录时发布尚未执行；后续成功收据不改写该历史条件。

统一最好仍 R15718/24，不重复本共同 harmonics 搜索，不追加旧预算。下一轮先实时查 R194+ 实际 source/output、完整命令、mainclean、资源；健康任务等待，不按旧 PID 操作。必要的新独立只读审计可从当前/因果 initial native34 与固定 nominal 重算 scalar/wave/request，再从 post IMU/right exact double 边界重算 reference±.18/merge/head/joint/slew/base/adjusted/applied/prev，并独立重算完整共同选择。先独立只读 smoke 包含标准/非零救回/退化/无效早退，自然退出再正式；仅 saved 值 compare，不用决定，不重复 R193 动态或加方向统计。预算≤.25GiB/reserve10GiB，唯一实际新编号和路径，不读未见320。

原门槛 24≥22且优于18/allvalid，之后316≥36/40且优于固定zero/R102/allvalid，再未见320两个不重叠20组各≥18/20、12秒进入/strict30/500Hzvalid；随后更大扰动、sensor/execution delay、右侧/prone/supine各20未见18/20全部valid。320仍未读执行。ACTIVE30分钟，仅实质进展/异常/需用户通知；全部仿真通过存证并列硬件未通过项才结束，不连接/使能/部署硬件、保护 live_server2664890、不改 walker。terminaltrue/fulltaskfalse/hardwarefalse/qualificationfalse。

