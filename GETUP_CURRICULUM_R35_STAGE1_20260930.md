# R35 curriculum stage 1: failed-boundary training, no verified gain

This experiment continued the stage-0 actor into a harder physical-reset
distribution, while keeping the original MuJoCo model, motor limits, 50 Hz
target slew, substep audits and strict loaded-home-standing criterion. It is
simulation only and **not** a real-robot controller.

Stage 1 completed 64 PPO iterations, 131,072 policy decisions and 655,000
actual motor controls. Unlike stage 0, the reset distribution includes the
first measured failing partial-tilt boundary for each direction. The final
training batch reported prone 3/7 one-second discoveries, supine 11/11,
left side 9/9 and right side 9/9. Four of the seven prone episodes violated
the unchanged physical audit. Training-batch success is not independent
validation and these are **not** fully fallen starts.

Deterministic 6-second validation used three held-out seeds per condition. The
final actor held a normal standing start in 3/3; first difficult partial-tilt
boundary: prone 0/3, supine 1/3, left side 3/3 and right side 3/3. These
results are essentially unchanged from stage 0. Actual fully fallen starts:
0/3 in **each** of prone, supine, left and right. The intermediate checkpoint
48 showed the same pattern. All failure trajectories, ONNX checkpoints,
training log, episode records and executed source snapshots are archived.

The next diagnostic uses exact same motor physics to test whether stronger
temporally correlated action exploration or short targeted pulses can find a
corrective trajectory across the prone/supine boundary. Merely training
longer on this unchanged distribution is not justified by the current
held-out evidence. Full completion still requires at least 18/20 independent
actual-fall recoveries **for every direction**, followed by 30 s continuous
loaded strict standing.
