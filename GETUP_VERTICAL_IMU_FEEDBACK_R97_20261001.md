# Open Duck vertical IMU feedback training

R97 tests the up-vector Z channel in early closed-loop recovery, after
R96 initial encoder compensation failed to improve the development baseline.
The existing four-feature feedback reads up-vector X/Y and scaled pitch/roll
rates, but not Z. This is a feature-conditioning hypothesis, not proof that
the omitted channel causes failures or that existing observations contain
no equivalent information.

## Controlled intervention

A separate physical nominal rollout records pre-control up-vector Z along
the frozen reference targets. Each evaluated trial reads its own current
IMU up-vector Z before the command and subtracts that reference. Three
gains produce opposite left/right hip-roll corrections and independent
neck-pitch and head-pitch corrections. The correction starts at time zero,
remains fully active through 1.2 seconds and fades to zero by 3.5 seconds.
This directly reaches the contact-sensitive early transition, unlike the
earlier X/Y feedback envelope starting at 0.5 seconds and reaching full
amplitude only at 1.5 seconds.

Original four-feature feedback, targets, timing, collision, mass, friction,
torque, joint range and target-speed limits stay frozen. Gains are bounded
to plus/minus two; old plus new feedback shares the original 0.18 rad cap.
Identity calls the literal baseline. All trials start with actual complete
settled falls and continue without state resets. Reference recording is a
separate physical run, never a live-trial state reset or injection. Runtime
reads no seed, outcome, root state, contact force or future observation.
The walking controller and hardware are not modified or enabled.

## Training and independent validation

The three-gain CEM search uses seed 197, six CPU workers, eighteen candidates
per generation, at most twenty-four generations and a twelve-stagnant-generation
limit. Development seeds are 769000 through 769007 and 773000 through 773015.
All evaluated nominal/development trajectories, vertical references, observed
Z-error arrays, physical audits and source/model hashes are retained. Search
and random-generator checkpoints are preserved. Identity must reproduce
nominal success and 13/24.

Only nominal-preserving count improvement enters the five-arm development
micro-command-bias comparison. Mean and worst counts must be no worse than
baseline and each baseline nominal success must be retained. This is
comparative development screening, not sensor/execution-delay qualification
or a claim that every biased nominal trial succeeds.

Only promotion runs fresh paired full-fall seeds 797000 through 797039,
with identical initial-state hashes and a thirty-five-second hold requiring
thirty uninterrupted strict-standing seconds. Paired failures, rescues and
regressions are retained. Otherwise these seeds stay unused. All original
four-posture gates, at least 18/20 unseen perturbations per posture, larger
independent sets and sensor/execution-delay tests remain required. One-second
development tails and training counts do not establish recovery success or
hardware readiness.

## Execution

Fifty-one code and regression tests passed before R97 launcher startup.
The actual main and six workers were subsequently observed with the expected
command, and generation one completed with best 13/24 and nominal success.
This is initial development evidence, not improvement or qualification.
Check current identities and growth before continuing; never act on old PIDs.
Source is `diagnostics/train_getup_vertical_imu_r97.py`;
output is `/data/shijinsheng/open_duck/outputs/getup_vertical_imu_r97_left_20261001`.
Do not resume completed R67 through R96 jobs or duplicate a healthy R97 job.
