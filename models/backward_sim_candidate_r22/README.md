# 固定平地仿真后退候选版本 R22

这是 **R2 参考残差模型 + R20 关节修正器 + 匹配的目标平滑/启停控制流程**，不是一个能直接替换 `BEST_WALK_ONNX_2.onnx` 的归一化动作模型。

- R2：`models/backward_reference_residual_r2/final.onnx`，SHA256 `05c1219a1403831deead152a6cff89b30a1d20fea3ebef285ca5c07279cfbdc8`。
- 修正器：本目录 `corrector.onnx`，输入 8 个预处理特征，输出 14 个关节角修正，单位为弧度。
- 解码：连续插值参考 dx −0.0925；残差增益 0.12；1 秒启动渐变；50 Hz 控制。
- 目标平滑：tau 0.01 秒，alpha = 0.02/(0.01+0.02)，先关节限位、再平滑、最后保留原电机速度限制；历史动作使用平滑后的限速前目标。
- 停车：用 1 秒 smoothstep 将后退关节目标平滑过渡到原站立策略；不要把零指令交给固定负参考解码器当作停车。

验证保持原标准：无倾倒，轴向后退速度 −0.10 至 −0.05 m/s，末段航向误差 ≤15°，航向 RMS ≤10°，侧向偏移 ≤0.25 m，最小机身 up-Z ≥0.94。组合动作还要求停车末尾 1 秒水平漂移 ≤5 mm。

已完成的独立测试：50 组×60 秒连续后退、20 组×120 秒连续后退，以及 50 组“站立3秒→后退10秒→停车3秒→重启后退10秒→停车3秒”，均达到对应标准。精确范围和校验信息见 `manifest.json` 与仓库 `results/`。

仅验证了固定平地、固定摩擦、标准站姿和 ±0.02 初始速度扰动。未验证真实电机、实际 IMU 噪声、通信延迟、外力扰动、坡面或真机。`hardware_readiness=false`；生产 Agent 没有自动换模。

在资源完整的服务器运行时仓库中复现组合测试：

```bash
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1 \
.venv/bin/python -m diagnostics.backward_skill_sequence \
  --contract /data/shijinsheng/open_duck/training/backward_sim_candidate_r22/controller_contract.json \
  --model /data/shijinsheng/open_duck/training/backward_sim_candidate_r22/corrector.onnx \
  --output /data/shijinsheng/open_duck/outputs/r22_user_sequence_test \
  --seed-start 1700 --seed-count 10 --workers 4 --stop-blend-s 1
```

输出目录必须尚不存在。GitHub 精简仓库不包含全部官方网格和参考动作资源，不能单独替代已搭建好的运行时环境。
