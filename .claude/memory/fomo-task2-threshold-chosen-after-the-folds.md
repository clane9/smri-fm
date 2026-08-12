---
name: fomo-task2-threshold-chosen-after-the-folds
description: How task 2 selects its hyperparameters — the threshold grid-searched on the out-of-fold probabilities rather than by an inner CV, why the resulting inflation is accepted, and why `C` is fixed where tasks 1 and 3 cross-validate it.
metadata:
  type: project
  observed: 2026-08-12
---

`main_task2.py` originally chose its threshold with a 5-fold inner CV inside every training fold
(2026-08-12 draft, Dice 0.187). Replaced on 2026-08-13 by a grid search over the out-of-fold
probabilities, run once after `leave_one_out`.

**Why the inner CV was not worth it.** It refit `fit_head` five extra times per outer fold — at
4.2s a fit, roughly 25s of each ~40s fold, about 5/6 of the fitting cost of a run — to produce a
threshold the draft's logs show was nearly constant across folds (4.7e-3 to 1.8e-2). The mechanism
it protected is bounded at +0.08 by the oracle decomposition (achieved 0.187 vs oracle 0.266).

**Why selecting on the scored subjects is accepted.** Connor's argument: the threshold was the
*only* hyperparameter given nested-CV treatment. `inverse_reg`, `modality` and the rest are all
tuned by reading the reported score and re-running, which is selection on the same held-out
subjects — just done wastefully, one full run per candidate. Grid-searching the threshold over
cached out-of-fold probabilities is the efficient version of something we were already doing.

The reported `dice` is therefore somewhat inflated relative to the leaderboard, by at most the
0.08 the oracle bounds. That is a known caveat, not a bug. **`dice` is still comparable to the
draft's 0.187 only approximately** — the draft selected out-of-fold, this does not.

I proposed a leave-one-out-within-the-out-of-fold selection to remove the inflation for free
(score subject *i* at the cut chosen on the other n-1, ~8 lines on the already-computed Dice
matrix). Connor rejected it as more machinery than the problem deserves, and cut it along with the
separate per-subject records. Worth remembering as a cheap option if the local number ever needs
to be defended rather than just compared against itself.

**Why `fit_head` fixes `C` where tasks 1 and 3 use `LogisticRegressionCV` / `RidgeCV`.** Connor
flagged the inconsistency in review. There, a row *is* a subject, so sklearn's internal split is
subject-level and the selection is legitimate. Here a row is a patch, thousands per subject, so an
internal split would put patches from the same subject on both sides of every fold — the selected
`C` would be chosen against a leaked score. `inverse_reg` is tuned from the config like any other
knob instead. Cost is also real: at ~300k weighted rows a single fit is 4.2s, so a 10-value path
inside every training fold is not free.

**The structural payoff.** Once the method returns probabilities rather than a mask, the score,
the oracle and the threshold search are all functions of `(probabilities, truth)`, so the protocol
computes them itself. That is what let the method interface shrink to `fit` / `predict_proba` /
`predict` / `threshold` — the protocol no longer touches `grid_size`, `patch_counts`,
`voxels_per_patch` or `cfg`. Full per-subject Dice-vs-threshold curves are saved to `curves.npz`
for post-hoc analysis instead of a chosen-point summary. See [[fomo-task2-seg-ideas-deferred]] for
what to spend the freed iteration budget on.
