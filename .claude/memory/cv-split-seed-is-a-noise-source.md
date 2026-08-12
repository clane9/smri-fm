---
name: cv-split-seed-is-a-noise-source
description: At n=48 the k-fold shuffle seed alone moved AUROC 0.948 -> 0.984, comparable to the tuning deltas being chased, which is why fomo_tune freezes the split seed independent of cfg.seed.
metadata:
  type: project
  observed: 2026-08-12
---

FOMO task 5, n=48, 20-fold, identical features and head: `random_state=4466` scored AUROC 0.948
[0.855, 1.000], `random_state=0` scored 0.984 [0.953, 1.000]. Nothing changed but which subjects
landed in which fold. Task 3 at n=494 barely moved under the same change (r 0.962 -> 0.963), so
this is an n effect rather than a task effect.

0.036 is larger than most method changes worth trying at this scale, so a split that moved with the
run's seed would let a tuning "win" be pure resampling. Hence `cross_validate` seeds its shuffle at
0 rather than from `cfg.seed` — the folds are protocol, not a knob. Connor made that change himself
after seeing the two numbers.

The uncomfortable corollary: the frozen split is one draw, and its score is one sample from that
spread. The bootstrap CI resamples subjects, not splits, so it does not contain this variance and
should not be read as if it does. Repeated CV would estimate it at k times the cost; the deliberate
choice is to hold the split and treat scores as comparable only within that fixed draw. Same
argument as `.claude/NOTES.md` thread 1, which is about the CI being wider than the deltas.
