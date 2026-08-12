---
name: verifying-a-submission-container
description: Three layers of check that a packaged .sif computes what cross-validation did, the two traps that cost time, and the question none of them answers.
metadata:
  type: reference
  observed: 2026-08-12
---

Run on the three FOMO containers (tasks 1, 3, 5) before the first leaderboard submission. Each
layer catches something the one above it cannot:

1. **The challenge's own validator.** Packaging, GPU visibility, output format. Weaker than it
   looks for regression: the task 3 suite is 12 tests against task 1's 20, and what it drops is
   the probability-range check — so nothing it runs constrains a regression output beyond
   finiteness. A head predicting a constant would pass all 12.
2. **Container vs host over every subject.** Load the model once inside the container, loop the
   whole cohort, diff against the same loop on the host. Came back `0.000e+00` across all 563
   subjects, which covers the stripped checkpoint, the pinned wheels and the `head.joblib`
   unpickle in a single number.
3. **One subject through the runscript.** `apptainer run` with `/input` and `/output` bound and
   the niftis *copied* in, which is how the challenge hands them over rather than how our loops
   read them. Matched to all six decimals the output format carries.

Two traps. **`data/fomo_eval` is a symlink** out of the repo, so `--bind <repo>:/repo` silently
yields a dangling link and a `FileNotFoundError` deep in the script; bind the resolved path too.
And **invoke `/app/predict.py` per subject only for the spot check** — it reloads a 1.3G checkpoint
every call, hours over a full cohort. Amortizing the load in one in-container process exercises the
same `Method.load` + `Method.predict` path for the sweep.

What none of it tells you is whether the score is *real*. Container and host read the same features
from the same cohort, so a confound passes through both identically and all three layers stay
green. That is a separate audit — see [[ppmi-mini-confound-floors]] for its shape.
