---
name: fomo-task2-seg-ideas-deferred
description: Four ways to push FOMO seg past a patch-level linear head — multi-block features, modality fusion, a linear sub-patch decoder, a conv head — deliberately deferred behind the baseline, with the argument for that order.
metadata:
  type: project
  observed: 2026-08-12
---

Considered and deliberately **not** in the task-2 baseline, in the order they are worth trying:

1. **Features from several blocks, not just the last.** Free — hooks on an existing forward pass —
   and for a frozen MAE the final block is usually worse for dense tasks than mid ones.
2. **dwi + flair fused.** The two share a grid after the transform, so their patch grids coincide
   exactly and fusion is a concatenation. Nearly free; left out only to keep the first number
   attributable.
3. **A linear sub-patch decoder: token -> 8x8x8 (or 4³ at 2mm) voxel logits.** `patch_embed` is a
   linear map over a 512-voxel block into 1024-d, so a token is overcomplete in its own block and
   early blocks retain most of that — within-patch structure is present, it just needs a head that
   emits more than one number per patch. Still one linear fit, still fits `Method.fit`/`predict`.
   **Demoted on 2026-08-12:** it was ranked here to buy resolution, and resolution turned out not
   to be the binding constraint (see the measurement below), so it should now sit behind (1) and
   (2) rather than being the headline fix.
4. **A conv head over interpolated features.** More machinery than (3) for the same target, and it
   replaces one closed-form fit with 23 optimisation runs each needing its own early stopping
   inside the fold.

**Why deferred:** at n=23 with per-subject Dice over lesions spanning 1000x in size, the SE on the
mean is ~0.03 and the CI is ~±0.07. That buys three or four *readable* comparisons on this task,
total, so they go to the cheap high-prior moves first. Interpolating patch features to 2mm was
rejected as a way to buy resolution on its own: a token is a summary of its 8mm cube, and the
interpolant only smooths it — (3) is the version of that idea with the information actually in it.

**Measured on the task-2 baseline (2026-08-12), and it moves the ordering.** Mean per-subject Dice
decomposes as ceiling **0.563** (best achievable on the 8mm patch grid, labels and geometry only) ->
oracle **0.266** (best global threshold on the model's own probabilities) -> achieved **0.187**. So
coarseness costs less than half of what feature quality costs: if the head ranked patches perfectly
by tumour fraction the oracle would *equal* the ceiling, making the 0.30 gap pure ranking quality.
The 0.08 below that is threshold selection, which bounds any operating-point work at +0.08. Even
the smallest tumour in the cohort (58 voxels) has a ceiling of 0.161.

**How to apply:** revive in that order, one at a time, each measured against the frozen protocol.
Anything under ~0.05 mean Dice cannot be distinguished from noise, so a change that does not have a
story for clearing that is not worth a run. See [[dice-ceiling-diagnostic-deferred]] for why raw
Dice stays the headline.
