# Open Duck early recovery timing results and running training

R75 found no improvement from its fifteen coarse early-motion timing
settings. R76 is already running a narrower bounded timing search on the
server; it has not completed independent qualification. This record separates
the finished training-only ablation from the still-running experiment.

## R75 completed timing comparison

Directory: `/data/shijinsheng/open_duck/outputs/getup_prefix_timing_probe_r75_20261001`.
The frozen identity path retains nominal success and 13/24 short training
successes. Global duration factors of 1.2 retain nominal success but achieve
only 4/24 training successes. All other thirteen settings fail the mandatory
nominal gate, so their printed zero is a gate rejection, not evidence that
all twenty-four disturbed falls were individually simulated and failed.
The selected factors are `[1, 1, 1, 1]`.

This result rejects improvement from the tested coarse timing settings. It
does not prove timing is irrelevant, establish a root cause for every failed
fall or justify relaxed physics. No unseen qualification seeds were used in R75.

## R76 observed running state

Directory: `/data/shijinsheng/open_duck/outputs/getup_prefix_timing_cem_r76_20261001`.
Server inspection found Python PID 1886896 and six workers, launched under
an execution lock. A duplicate launch was refused and no existing process
was stopped. The observed log contained four completed generations, retaining
13/24 short training successes and nominal success. A checkpoint and search
history existed; there was no completed independent report at inspection.

The running command uses the R73 zero-correction checkpoint and R75's identity
selection, seed 176, 32-generation budget, population 24 and six workers.
The source applies twelve-stale stopping and then paired independent complete
fall qualification on seeds 778000 through 778039. Qualification requires
at least 30 continuous strict-standing seconds during a 35-second final hold,
with the original collision, torque, joint, target-slew and standing gates.
There is no mid-rollout root reset or teleportation. Earlier/later walking
controllers and real-robot programs remain untouched.

## Verification and continuation

Eighteen unit tests passed on the simulation server, covering identity timing,
resampling bounds and endpoints, isolated intervention scope, existing limits,
sensor reference timing and the continuous-standing gate. These tests verify
code contracts, not physical get-up competence. Source and dependency hashes
are retained in the experiment contracts; final checks must compare them to
the actually evaluated files.

Next inspect R76's actual processes, log and final report before starting any
other experiment. Preserve failures and paired trajectories. Do not describe
the left-side skill as reliable until independent qualification passes; right
side, prone, supine and delay/noise robustness remain pending. No deployment
or real-robot connection is authorized by this simulation continuation.
