# 后退命令课程训练记录（2026-09-27）

## 目的

v30-v33 已经能在 5 个扰动种子中保持站立，但在 `-0.074 m/s` 命令下速度约为 `-0.025~-0.029 m/s`，并且累计偏航约 65--83 度。本轮不再重复简单的速度/接触奖励扫描，而是：

1. 先做原始/左右镜像/相位翻转的最小闭环诊断；
2. 给累计航向误差增加预热和渐增机制，避免像 v31 一样突然加入强航向约束；
3. 从稳定的 v30 检查点做 `-0.035 -> -0.050 m/s` 命令课程；
4. 对 `BACKWARD_HEADING_ERROR=4` 与 `8` 做单变量对照。

训练源码目录：`/data/shijinsheng/open_duck/projects/Open_Duck_Playground`  
恢复检查点：`/data/shijinsheng/open_duck/training/backward_contact_step_v30/2026_09_27_182859_4259840`  
训练 GPU：GPU6（训练期间无其他训练进程）。

## 代码变更

`playground/open_duck_mini_v2/focused_skill.py` 新增累计航向误差的分阶段系数：

- `BACKWARD_HEADING_WARMUP_STEPS`：预热步数，默认 `0`；
- `BACKWARD_HEADING_RAMP_STEPS`：渐增步数，默认 `1`；
- 系数在预热后从 0 线性增加到 1；
- 默认值保持既有行为不变。

`scripts/launch_backward_curriculum.sh` 记录 GPU、命令、恢复检查点和所有 `BACKWARD_*` 变量，并支持从上一阶段的检查点继续训练。

## 最小诊断

模型：v30 `final.onnx`。固定 MuJoCo 状态，命令航向为 `-0.074`，每个模式运行 5 秒：

| 模式 | 完成时间 | 航向变化 | 最小 up-z | 摔倒 |
|---|---:|---:|---:|---|
| 原始 | 5.0 s | -5.91° | 0.99047 | 否 |
| 左右镜像，同相位 | 5.0 s | -4.08° | 0.99417 | 否 |
| 左右镜像，相位翻转 | 5.0 s | -4.17° | 0.99396 | 否 |

结论：在固定状态下没有发现一个简单的镜像/相位变换就会立即破坏闭环稳定性；主要问题仍然是长时间动力学、足端支撑和奖励目标之间的耦合。

结果文件：`results/diag_backward_equivariance_20260927.json`。

## 训练与 5×10 秒验收

验收条件：5 个 `qvel` 扰动种子、每个 10 秒、无摔倒；同时记录后退速度、横向位移和航向变化。当前项目的部署门槛仍为平均速度绝对值至少 `0.037 m/s`，且需要低偏航和低横向漂移。

| 模型 | 命令 | 完成 | 平均后退速度 | 平均横向位移 | 航向变化（逐种子） |
|---|---:|---:|---:|---:|---|
| `backward_curriculum_s1` | -0.035 | 5/5 | -0.022553 | 0.129496 | -32.4°, -29.9°, -32.8°, -30.3°, -34.8° |
| `backward_curriculum_s2` | -0.050 | 5/5 | -0.025920 | 0.241008 | -52.6°, -54.2°, -56.3°, -58.0°, -63.5° |
| `backward_curriculum_s1_heading8` | -0.035 | 5/5 | -0.024357 | 0.171346 | -36.7°, -42.9°, -48.4°, -46.9°, -45.0° |

结果文件：

- `results/sus_backward_curriculum_s1.json`
- `results/sus_backward_curriculum_s2.json`
- `results/sus_backward_curriculum_s1_heading8.json`

训练日志和模型：

- `results/backward_curriculum_s1_train.log`
- `results/backward_curriculum_s2_train.log`
- `results/backward_curriculum_s1_heading8_train.log`
- `models/backward_curriculum_s1_final.onnx`
- `models/backward_curriculum_s2_final.onnx`
- `models/backward_curriculum_s1_heading8_final.onnx`

SHA-256：

- s1: `f23fe6c23af712da6cee8beae9e9422d8e295a36c3fe54b95e803eb7273295a3`
- s2: `32c7ce51f7fc1bcefa75c828ce773b62e690ce48a2c6aca5147819b85e0e2f7d`
- s1_heading8: `a3abc059df3f1aacb5b0f23a44f7bf73044825e7ee793c0174750929ba107`

## 结论

本轮没有达到部署门槛：三组均能保持站立，但平均速度仍低于 `0.037 m/s`；提高命令到 `-0.050` 后速度没有增加，横向位移和偏航反而明显变差；把累计航向误差权重从 4 提高到 8 也没有改善。因此不继续把命令推到 `-0.074`，也不推荐任何本轮模型上真机。

下一步不应继续扫描单个标量奖励。应基于足端落点、左右支撑时序和步幅构造更明确的长步幅接触参考，并继续要求 5×10 秒稳定验收。真机测试还必须经过已有的离线回放、安全层、50 Hz 时序和人工监督检查。
