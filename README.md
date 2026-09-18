# Open Duck Mini 仿真训练与大模型控制

这是基于 [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground) 的实验代码快照，包含：

- `stand / walk / turn / stop` 连续控制环境与 PPO 训练入口；
- 后退、侧移、弧线和转向等专项课程；
- 后退策略的 JAX/MJX 与原生 MuJoCo 双重验证；
- 统一动作工具接口、MiniCPM-o 本地规划器和实时网页控制服务。

本仓库不包含 MuJoCo 机器人资源、官方 ONNX、训练 checkpoint 或本地大模型。它是一个可覆盖到上游仓库的代码层，避免重复上传大文件。

## 1. 准备上游工程

```bash
export OPEN_DUCK_ROOT="${OPEN_DUCK_ROOT:-$HOME/open_duck}"
mkdir -p "$OPEN_DUCK_ROOT/projects" "$OPEN_DUCK_ROOT/training" "$OPEN_DUCK_ROOT/outputs"

git clone https://github.com/apirrone/Open_Duck_Playground.git \
  "$OPEN_DUCK_ROOT/projects/Open_Duck_Playground"
git -C "$OPEN_DUCK_ROOT/projects/Open_Duck_Playground" checkout b9be205ac64488c23504ca42e5ec790337adeec3

git clone https://github.com/ruimengmeng9-spec/openduck-mini-.git \
  "$OPEN_DUCK_ROOT/projects/openduck-mini-training"
cp -a "$OPEN_DUCK_ROOT/projects/openduck-mini-training/playground/." \
  "$OPEN_DUCK_ROOT/projects/Open_Duck_Playground/playground/"
cp -a "$OPEN_DUCK_ROOT/projects/openduck-mini-training/open_duck_agent" \
  "$OPEN_DUCK_ROOT/projects/Open_Duck_Playground/"
cp "$OPEN_DUCK_ROOT/projects/openduck-mini-training/"*.py \
  "$OPEN_DUCK_ROOT/projects/Open_Duck_Playground/"
```

当前实验环境使用 Python 3.11、JAX 0.10.2、MuJoCo/MJX 3.12.0、Brax 0.14.2 和 `playground==0.0.3`。较新的 `playground` 包缺少本项目使用的旧接口。

## 2. 关键兼容修改

`playground/open_duck_mini_v2/joystick.py` 包含两项必要修改：

1. 与普通 MuJoCo 推理保持一致，在加速度计 X 轴观测中加入 `+1.3` 偏置；
2. 增加可配置的 `reward_floor`。普通任务仍默认为 0，后退专项训练允许负奖励，避免“站着不动”进入零梯度区域。

`focused_skill.py` 定义专项指令分布、奖励项和安全终止条件；`focused_skill_runner.py` 负责 PPO 训练、Orbax checkpoint 和 ONNX 导出。

## 3. 启动后退专项训练

V13 使用低速课程（约 `-0.025` 到 `-0.035 m/s`）、归一化进度奖励、紧超速限制和提前倾倒终止。建议只暴露一张空闲 GPU：

```bash
cd "$OPEN_DUCK_ROOT/projects/Open_Duck_Playground"
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export TF_FORCE_GPU_ALLOW_GROWTH=true

.venv/bin/python -u -m playground.open_duck_mini_v2.focused_skill_runner \
  --skill backward \
  --output_dir "$OPEN_DUCK_ROOT/training/backward_v13" \
  --num_timesteps 12000000 \
  --num_envs 1024 \
  --num_evals 7 \
  --num_eval_envs 64 \
  --seed 56 \
  --learning_rate 2e-5 \
  --restore_checkpoint_path /path/to/a/reverse-motion/checkpoint
```

也可以使用 `scripts/train_backward.sh`，通过环境变量 `RESTORE_CHECKPOINT` 指定恢复点。

## 4. 验证策略

训练回报不能代替动作验证。至少执行以下两条链路：

```bash
# 直接在 JAX/MJX 训练环境中检查 checkpoint
.venv/bin/python evaluate_jax_backward.py /path/to/checkpoint --speed -0.04 --steps 250

# 将候选 checkpoint 导出成 ONNX 后，在普通 MuJoCo 运行时验证
.venv/bin/python validate_backward_v11.py

# 比较转向候选与已部署转向策略的实测转速
.venv/bin/python validate_turn_v2.py
```

接受一个后退模型前，应检查：

- 至少连续 5 秒不触发终止；
- `up_vector_z > 0.5`，机身高度不低于 0.08 m；
- 局部前向速度为负，且能随负向指令变化；
- 多随机种子和启停切换下结果一致。

接受一个转向模型前，应检查：

- 正负指令的实测偏航角速度都随指令单调变化；
- 两个方向的响应幅度接近，不再出现单向偏弱；
- 5 秒测试内 `up_vector_z` 保持接近 1 且不触发跌倒。

## 5. 动作接口与大模型规划

先运行不依赖大模型的测试和演示：

```bash
.venv/bin/python -m unittest open_duck_agent.test_tools open_duck_agent.test_agent -v
.venv/bin/python -m open_duck_agent.agent --demo
```

`open_duck_agent` 将自然语言计划限制为 `stand / walk / turn / stop` 等安全工具调用。本地 MiniCPM-o 的配置与实时页面启动方法见 [open_duck_agent/README.md](open_duck_agent/README.md)。

## 6. 当前实验结论

- 官方预训练策略已经通过 30 秒、多随机种子标准站姿验证；
- 站立、直走、停止、重新起步均可用；
- V11 后退策略 15/15 组完成 5 秒安全验证，但速度约 `-0.006 m/s`，未达到部署标准；
- V12 已产生明显后退（约 `-0.056` 到 `-0.097 m/s`），但会在 2–4 秒内后仰倾倒，未部署；
- V13 已收紧目标速度、倾角、机身高度和超速约束，部署模型仍保留已验证基线；
- Turn V2 把 `-0.3 rad/s` 指令下的实测转速从 `-0.127` 提升到 `-0.207 rad/s`，
  四组正负测试均 5 秒不跌倒，负向偏弱问题已缓解；提升幅度见
  [results/turn_v1_vs_v2_20260918.json](results/turn_v1_vs_v2_20260918.json)。

详细实验演进见 [TRAINING_NOTES.md](TRAINING_NOTES.md)。
