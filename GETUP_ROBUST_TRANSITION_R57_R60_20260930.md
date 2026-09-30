# Get-up robust-transition experiments R57–R60 (2026-09-30)

## Scope

These experiments are simulation-only. They start from the physically reached
two-foot transition state produced by the actual left-side fallen replay. They
do not yet validate other fall orientations, the entire perturbed fallen path,
or hardware deployment. Scene physics, contacts, actuator limits, target slew,
and the strict loaded-standing gate are unchanged.

## R57: transition domain randomization

`search_getup_robust_transition_r57.py` perturbs orientation (±0.018 rad), all
actuated joint positions (±0.008 rad), and generalized velocity (±0.008) at the
captured R49 transition. CEM optimizes the lower-performing cases rather than a
single nominal trajectory.

- Training: 6/6 in 26 generations.
- Independent validation: 11/20, up from the R49 baseline of 1/20 on the same
  seeds.
- Canonical actual-fall replay remained successful.

## R58: structured IMU residual feedback

`search_getup_feedback_balance_r58.py` adds eight structured tilt/rate gains to
the R57 target sequence. It reached 8/8 on its training perturbations, but
degraded a fresh validation group from 12/20 to 9/20. This is retained as a
negative result and is not the selected controller.

## R59: active hard-case refinement

`search_getup_hardcase_refine_r59.py` mixes the nine failed R57 boundary seeds
with three successful seeds and optimizes the mean of the five worst physical
scores, plus the strict-success count.

- Training: 12/12 in 13 generations.
- Forty new, unseen transition perturbations: 33/40 (82.5%).
- Baseline R57 on those same forty seeds: 25/40 (62.5%).
- All evaluated training candidates selected as the winner remained within the
  existing physical audit limits.

## R60: long-hold verification

`validate_getup_hardcase_long_r60.py` replays the forty R59 held-out
perturbations and requires a 35-second home-target hold with at least 30 seconds
of continuous strict standing at the end.

- Long-hold result: 33/40 (82.5%).
- Successful trials had about 34.7 seconds of continuous strict standing.
- Canonical actual-fall replay: 34.68 seconds continuous strict standing,
  physically valid, final support margin about 0.0283 m.

## R61: second hard-case round

`search_getup_hardcase_round2_r61.py` feeds the seven R60 failures back into a
second search while retaining five successful states. It again reached 12/12
on that fixed training set, but a third, unseen forty-seed group remained
30/40 before and after training. This confirms that further fixed open-loop
hard-case rounds are overfitting rather than improving generalization. R61 is
archived as a negative result and does not replace R59.

## Current conclusion

R59 is the selected result of this group. It materially improves transition
robustness and passes long-hold verification for every short-hold success, but
seven of forty unseen transition perturbations still fall. Therefore the full
get-up task is **not complete**, `hardware_readiness` remains false, and this
checkpoint must not be connected to the real robot yet. The next stage should
train a richer phase-aware feedback or residual policy and then repeat full
fallen-state, multi-orientation, long-hold, and hardware-safety validation.
