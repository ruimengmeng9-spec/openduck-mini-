# 独立起身技能开发：2026-09-28

## 当前状态

已经建立独立的倒地接触模型、参考动作搜索、站立策略交接、严格评估和轨迹录像流程。**尚未得到通过验证的起身策略，不可部署真机，也未向语言控制器暴露 getup 动作。** 这轮优化的是关节参考轨迹，并非已经完成神经网络 PPO/蒸馏训练。

前进、转向、后退和停车候选模型、原始场景均未替换。本次生成的场景和参考动作只放在独立实验目录。

## 本轮纠正的事项

1. 旧评估中的方向标签和旋转符号存在重复：趴倒/仰倒、左侧/右侧实际上会得到重复旋转。新代码明确采用身体 +X 向前、+Y 向左、+Z 向上的四种不同姿态。
2. 旧 getup 沿用行走策略的 `home + 0.25 * action`。tanh 输出限制在 [-1,1] 时，目标关节只覆盖站姿附近 ±0.25 rad。新起身接口可以使用原有物理关节范围内的绝对目标，不增加机械限位或电机能力。
3. 原始场景主要有脚底碰撞；旧起身场景只补了躯干/头部盒体。新探索场景给外观网格启用对地碰撞，补足腿部接触。模型的质量、惯量、关节限位、阻尼、摩擦和电机参数数组逐项检查保持一致。
4. 三帧动作历史独立移位，传感器按地址/维度取值，分支状态恢复包含控制目标和求解器 warm-start。
5. 不以单帧“朝上”或高度作为成功。需要从真实倒地起点执行参考动作、接入原站立策略，最终持续至少 2 s 满足朝上、正常高度、有足部支撑、头/躯干离地且速度较低的条件。

摩擦系数保持原始场景的 0.6，物理步长 0.002 s、控制周期 0.02 s、电机指令变化速度上限 5.24 rad/s、原电机力矩范围 ±3.23 Nm。没有在执行过程中改写机身位姿/速度、没有施加人工扶正外力。

## 已运行的三个方向

| 版本 | 方法 | 目的 |
| --- | --- | --- |
| R1 | 双侧对称关节参考 + CEM，16 代 × 64 候选/姿态 | 检查扩展到原机械范围后能否撑起 |
| R2 | 独立左右腿参考 + 稠密进度奖励 + 状态门控交接，24 代 × 96 候选/姿态 | 允许翻身/非对称支撑，避免固定时间动作破坏已获得的支撑 |
| R3 | 保存真实可达中间状态的运动原语束搜索，最多 7 层、8 条路径 | 分阶段搜索翻身、撑起和交接路径 |

每个版本搜索趴倒、仰倒，保存最优参考与失败轨迹，并各进行 20 组扰动起点验证。当前三种搜索方法都未获得连续站稳的成功案例；严格复核以各目录的 `verified_results.json` 为准。初始环境还检查了左右侧倒，但本轮未声称侧倒技能已训练。

正常站姿对照可以保持约 9 s 稳定站立，说明新接触场景并非连正常站立都无法维持。这个对照不能证明倒地碰撞几何准确，也不能证明起身失败仅由优化算法造成。

## 尚未解决的建模门槛

当前对地碰撞使用原外观网格的凸包，没有完成可用于硬件安全判断的自身碰撞校准。直接把全部网格用于自身碰撞，在正常站姿就筛出头部/连接件、髋部/躯干的重叠，最大凸包重叠达到厘米级；这可能包含中空零件的凸包误报，不能据此认定真实零件一定穿透，也不能简单忽略后宣布安全。

MuJoCo 的网格碰撞使用凸包，非凸/中空结构需要合适的碰撞几何或分解，参见 [官方碰撞说明](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection)。本轮独立搜索是寻找动作线索，不是最终真机物理验证。

未搜索到成功动作也不能证明机械结构不可能起身；当前的关键待查项是足部可达支撑位置、头/躯干几何、可用关节行程和中间姿态的力矩/重心条件。

## 文件和复现

- `diagnostics/getup_independent_native.py`：独立模型、姿态、绝对关节目标、CEM R1、严格评估。
- `diagnostics/getup_feedback_reference.py`：R2 状态门控参考搜索、录像。
- `diagnostics/getup_beam_reference.py`：R3 真实可达状态束搜索和连续回放。
- `diagnostics/audit_getup_self_collision.py`：凸包自身碰撞筛查，不修改训练场景。
- `diagnostics/validate_getup_independent.py`：保存候选的重新验证，保存当前代码、场景、模型哈希。
- `diagnostics/render_getup_verified.py`：复核轨迹录像，包含失败候选，不能视作成功演示。
- `diagnostics/test_getup_independent_native.py`：六项测试覆盖姿态、动作范围、传感器地址、历史和安全交接。
- `scripts/launch_getup_independent_r1.sh`、`r2.sh`、`r3.sh`：服务器独立搜索入口，不重复使用既有实验输出目录。
- `results/getup_independent_20260928/`：小型参考文件、搜索/复核结果、轨迹及接触审计。

运行资源完整的服务器 checkout，先加载现有环境：

```bash
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  .venv/bin/python -m unittest diagnostics.test_getup_independent_native -v

CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  .venv/bin/python -m diagnostics.validate_getup_independent \
  --experiment /data/shijinsheng/open_duck/training/getup_independent_r2_supine
```

场景生成器支持 `--root`，按该目录下的官方模型资源重新生成场景；已归档 XML 中的服务器绝对资源路径不可直接在另一台机器照搬。

## 下一阶段

先校准接触/自身碰撞几何并量化足部支撑可达性；寻找能从倒地进入可站立区域的成功参考。参考通过扰动起点、连续站立和策略交接验证后，才启动独立起身网络的行为克隆/PPO 训练，并进一步扩大侧倒、不同关节姿态、延迟和传感器噪声测试。不得蒸馏本轮失败参考并把其标记为已学会起身。
