# Open Duck Mini v2：转向 v8 与后退 v28 验证（2026-09-26）

本次所有运动测试均在服务器的原生 MuJoCo 平地仿真中完成，未向真机发送控制命令。先保留 v5 主策略及 v6/v7 转向模型，不覆盖历史产物。

## 转向阶段结论

- 以 v5 为基础，在负转向命令上使用 v8 ONNX 残差；v8 将较低负角速度和机身质量 ±10% 的训练样本纳入数据集。训练集 62,250、验证集 15,750 条；模型 SHA256 为 `a5783314208fddb979bb80f5d9cf16b34b6af00f1658deb6980dcbb22a1f0f37`。
- 独立九阶段测试（站立、慢速前进、停止、前进、停止、正转、停止、负转、停止）在 −0.10、−0.15、−0.17 rad/s 三档共 **40 组 / 360 阶段完成，0 次跌倒**。5 秒负转向平均转角分别为 −32.11°、−43.97°、−45.99°；对应理想角度约为 −28.65°、−42.97°、−48.70°。这说明转向已能稳定按方向响应，但不是精确速度跟踪。
- −0.15 rad/s、机身质量缩放 0.9/1.1 的各 5 组也均完成；平均转角分别为 −47.66°/−40.53°，显示模型仍对质量误差敏感。此前无电机目标速度限制的 5 组只测试过 v7，不能推论 v8 或真机已经安全。
- 当前九阶段测试里的 +0.03 m/s “慢速前进”几乎没有净移动。因此这套基线不能称为全速度范围都已就绪，更不能据此直接上真机。

相关数据：[`results/negative_mirror_distill_v8_training_summary.json`](results/negative_mirror_distill_v8_training_summary.json)、[`results/negative_mirror_distill_v8_final_m010.json`](results/negative_mirror_distill_v8_final_m010.json)、[`results/negative_mirror_distill_v8_final_m015.json`](results/negative_mirror_distill_v8_final_m015.json)、[`results/negative_mirror_distill_v8_final_m017.json`](results/negative_mirror_distill_v8_final_m017.json)、[`results/negative_mirror_distill_v8_mass0p9_m015.json`](results/negative_mirror_distill_v8_mass0p9_m015.json)、[`results/negative_mirror_distill_v8_mass1p1_m015.json`](results/negative_mirror_distill_v8_mass1p1_m015.json)。

## 后退阶段结论

此前 v12–v27 候选存在“后退但约 2 秒跌倒”和“稳定但几乎不动”两种退化。先测 v5 主策略在 −0.074 m/s、10 秒、5 个扰动种子：5/5 不跌倒，但平均前向速度为 +0.000037 m/s，即没有后退。

复查参考步态后发现，按 `action_scale=0.25` 换算，有 **21.4% 的参考动作分量超出 [-1, 1]**，旧版 v21 行为克隆把这些目标裁剪了。本次训练了不裁剪目标、线性输出的 v28 参考模型，共 27 个相位样本、8,000 轮；拟合 MSE `6.40e-7`，最大动作误差 0.00369，目标动作最大绝对值 2.358。模型路径为 `/data/shijinsheng/open_duck/training/backward_reference_bc_v28_unbounded/final.onnx`，SHA256 `899f0ead3c9f8c244417628222e3af77104e4b338e896f01a03a04cdcea907f2`。

但闭环结果不合格：从正常 3 秒站立热身切入，5/5 在 10 秒测试内跌倒，虽短时平均后退约 −0.079 m/s；从 home 姿态且关闭电机目标速度限制启动，5/5 均站立 10 秒，但平均速度为 **+0.00248 m/s** 且偏航约 63°，并非有效后退。故“参考轨迹不倒”不能当作“稳定后退步态”，v28 只保留为诊断模型。

随后测试了 v5 稳定策略与 v26 后退策略的动作混合。在 −0.074 m/s、10 秒、每档 5 个扰动种子下，v26 权重 0.25–0.90 的候选均 5/5 不倒，但后退仅约 −0.0006 至 −0.0035 m/s；权重 0.92、0.93、0.94、0.95 的完成数分别为 0/5、0/5、1/5、1/5，虽然跌倒前有约 −0.06 m/s 的后退。这里出现明显的速度—稳定性断崖，**没有找到既能有效后退又能持续站稳的候选**。

相关数据：[`results/backward_reference_bc_v28_training_summary.json`](results/backward_reference_bc_v28_training_summary.json)、[`results/backward_v5_baseline_20260926.json`](results/backward_v5_baseline_20260926.json)、[`results/backward_v28_unbounded_10s_20260926.json`](results/backward_v28_unbounded_10s_20260926.json)、[`results/backward_v28_home_no_slew_20260926.json`](results/backward_v28_home_no_slew_20260926.json)、各 `results/backward_blend_v26_a*_20260926.json`。

## 下一步

不再把 v28 或高权重混合策略作为可部署后退技能。先记录并分析 v26 在跌倒前的姿态、足底接触和电机目标变化，确认是否存在可通过相位、落脚或反馈控制解决的共同失稳模式；再决定是否针对该模式训练新的闭环后退策略。真机测试仍需满足 [`REAL_ROBOT_TEST_GATE_20260926.md`](REAL_ROBOT_TEST_GATE_20260926.md) 的独立安全条件。
