# Open Duck head command tolerance diagnostic

R89 teacher distillation and R90 nominal-output centering did not improve
the baseline. R91 now quantifies the sensitivity of the frozen get-up path
to tiny head-command errors. The purpose is to determine whether the
observed small imitation error is already sufficient to alter recovery
outcomes, before choosing another training objective or trajectory method.
This experiment is diagnostic, not a model-promotion search.

## Evidence motivating the test

R89 reproduced 25/25 case-specific short-gate teacher successes but achieved
at most 9/24 with a common student, despite small fitting loss. R90 preserved
the nominal trajectory exactly with centered feedback but reached at most
7/24 on perturbed development cases. These facts rule out nominal imitation
bias as a sufficient single-factor fix; they do not prove that every failure
has the same cause or that all feedback learning methods will fail.

## Prescribed physical comparison

Seventeen settings contain identity and separate positive/negative neck-pitch
or head-pitch raw biases at 0.0005, 0.001, 0.002 and 0.005 radians. They pass
through the same early activation envelope and combined 0.18-radian residual
cap. Even the largest prescribed bias is less than 0.29 degrees. All
reference joint targets, original leg-feedback law, phase timing, settling,
physics, joint and torque limits, target slew and strict-standing definitions
are frozen. No increase in allowable actuator commands is introduced.

Each setting is run from nominal and the same twenty-four actually settled
fallen development starts, with six CPU workers. Identity must reproduce
nominal success and 13/24. Initial-state hashes, physical audit, successes,
rescues and regressions relative to identity, and complete traces for both
successful and failed trials are saved. A nominal failure remains visible
rather than hiding sensitivity by skipping that arm's perturbed cases.
No intermediate state reset, hardware connection or original walking-control
edit is allowed.

The one-second strict development tail is only a diagnostic comparison. No
new independent seeds or thirty-second qualification are used, no candidate
is promoted directly, and no four-pose gate or hardware-readiness claim is
made. A favorable small-bias setting would require later training and fresh
full-fall qualification, not deployment.

Twenty-four source-contract tests passed before launch. Wrapper PID 2076760
was observed at launch; verify current identity before any process operation.
R67 through R90 are complete and must not be restarted.

Output: `/data/shijinsheng/open_duck/outputs/getup_head_tolerance_r91_left_20261001`.
Source: `diagnostics/probe_getup_head_tolerance_r91.py`.
Launcher: `scripts/launch_getup_head_tolerance_r91_20261001.sh`.

After completion, inspect result counts together with early trajectory and
IMU differences, physical rejection and command error. Distinguish a narrow
recovery basin from student distribution shift or insufficient observation;
do not merely repeat the failed fixed-gain or biased-distillation search.

## Completed sensitivity result

All seventeen prescribed settings finished. Identity reproduced nominal
success and 13/24. Only three of sixteen nonzero settings preserved nominal
success. No nonzero setting exceeded thirteen development recoveries;
nonzero settings had twenty through twenty-four physically valid cases.
For neck biases minus/plus 0.0005 radians, recovery fell to 6/24 and 8/24,
respectively, with eleven and nine baseline-success regressions. A minus
0.0005-radian head-pitch bias retained the aggregate 13/24 but rescued six
cases and lost six others, while failing nominal. Equal aggregate count
therefore does not imply equal reliability.

These comparisons support sensitivity to very small commanded perturbations
as one obstacle to transferring a fitted feedback law. They do not isolate
the underlying contact transition or establish a universal hardware cause.
No new common recovery policy passed, and no independent qualification or
hardware trial was performed. R91 is complete; no get-up process remained
at the final process check. Continue with saved failure analysis and a
clearly distinct robustness hypothesis, not a duplicate sensitivity run.
