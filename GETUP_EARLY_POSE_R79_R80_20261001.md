# R79 failure evidence and R80 early recovery training

Simulation only. No real-robot access, motor enable, deployment or changes to
the existing walking controller are part of this continuation.

## Latest state (2026-10-01, Asia/Shanghai)

R76--R78 are finished, not paused or still running. R76 retained identity
timing and qualified 13/40, identical to its paired baseline. R77's best
fixed wait remains 13/24 short training successes; its retrospective union
is 16/24, not a deployable policy. R78's best leave-one-case-out selector
remains 13/24. Do not restart these completed searches.

R79 is an offline audit of the recorded R77 training trajectories. It does
not create new physical rollouts, reconstruct contact forces or establish
new physical success. Eight training cases fail under every eligible wait.
Under the 2 s baseline, these eight reach maximum body up-Z values only
0.29--0.37. Six show a persistent reference-tilt mismatch starting around
1.94--2.50 s, before the home wait; the remaining two diverge later.
Waiting longer or shorter cannot by itself correct an already different
body/contact configuration. This is evidence for an intervention test, not
proof of a unique cause.

## R80 controlled hypothesis

The R73 pose search only altered the final two seconds of the 6.16 s early
recovery (approximately 4.16--6.16 s), after the earliest divergence. R80
instead learns two smooth leg-target correction knots at 1.5 and 2.5 s.
Their envelope is zero at 0.5 and 3.5 s and everywhere outside that window.
The eight leg joints use at most 0.20 rad target corrections, clipped to the
existing joint limits and executed through the unchanged 50 Hz slew limiter.
This bound is a policy search bound, not a relaxation of any physical limit.

Frozen: head targets, all targets outside the early window, phase timing,
2 s home wait, later pulse and terminal commands, feedback gains, collision
model, torque limits, joint ranges, target velocity limits and strict gates.
Nominal sensor references are regenerated for each candidate's own target
stream; reference states are never injected into a perturbed physical trial.
Each complete rollout starts from its own physically settled fallen state.
There is no intermediate root reset or teleport.

Training cases: 769000--769007 and 773000--773015, with a mandatory nominal
preservation gate. The zero-delta candidate must reproduce the R77 13/24
short baseline before training may proceed. Six CPU workers, fixed search
seed 180, population 24, at most 24 generations and twelve stale generations.
Nineteen unit tests passed before launch; they check code contracts, not
successful recovery. Training checkpoints, search history, RNG state,
executed source snapshots and source/model hashes are saved.

After fitting, the selected candidate and frozen baseline are tested on
forty new seeds 780000--780039, plus nominal. These seeds are excluded from
fitting. Full rollout completion, original physical audit validity and at
least thirty continuous strict-standing seconds in a 35 s final hold are
required. Baseline and candidate failure trajectories are both preserved.
Training success or reward alone is not qualification.

At this archive R80 has been launched and its main process and six workers
are live. No completed generation or qualification result is claimed here.
Output: `/data/shijinsheng/open_duck/outputs/getup_early_pose_r80_left_20261001`.
Entrypoint: `diagnostics/train_getup_early_pose_r80.py`.
Launcher: `scripts/launch_getup_early_pose_r80_20261001.sh`.

## Continuation

Read this note and R80's actual log/checkpoint/results before relying on an
older heartbeat prompt. Do not duplicate a live job. If it completes, inspect
paired rescues, regressions and physical failures before choosing another
method. A negative result is to be archived, not concealed or fixed by
loosening gates. The left-side stage remains unqualified; right-side, prone
and supine recovery, expanded unseen sets and sensor/execution delays remain
unpassed. No hardware readiness is established.
