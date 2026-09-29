# R35 stage-2 reset-audit regression and retry

This remains simulation-only Open Duck Mini v2 get-up research. It does not
replace or authorize any real-robot controller.

The first stage-2 PPO run started from the stage-1 actor and mixed 20% actual
fully fallen resets into harder partial-tilt resets. It stopped after two
iterations (4,096 policy decisions) with `Curriculum reset failed physical
audit`. Its eight completed actual-fall episodes yielded zero one-second
discoveries. The failed run, launcher log and executed-source snapshot are
preserved under `results/getup_curriculum_r35_stage2_failed_20260930`.

The failure was reproducible without PPO: set the audit's prior-episode
`floor` peak to 0.02 m, then request a new full-fall reset. The newly prepared
supine state had only about 0.000017 m instantaneous floor penetration, yet
`physical_valid()` failed because the accumulated peak remained 0.02 m.
`AuditedSim.prepare()` resets MuJoCo state but does not clear audit peaks;
partial-tilt preparation already cleared them before settling, while the
full-fall branch did not. Thus a physically invalid *previous* episode could
incorrectly invalidate the next reset and kill all PPO workers.

The full-fall branch now clears the prior audit **before** its 40 settling
motor controls, then audits those controls and checks the resulting state
with the unchanged physical thresholds. It does not suppress new violations.
The regression test proves both that stale peaks no longer reject a valid
reset and that a new invalid peak is still rejected. A native-MuJoCo
reproduction and four curriculum unit tests passed. The retry starts from
the same stage-1 final actor; the failed run was not silently resumed or
counted as a successful experiment.

The stage-2 retry result and held-out actual-fall validation must be appended
separately. The final goal is still unproven: each of prone, supine, left and
right must meet at least 18/20 independent actual-fall recoveries and then
hold the strict loaded standing condition for 30 seconds.
