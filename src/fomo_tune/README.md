# fomo_tune

The five FOMO26 challenge tasks, one script each, tuned independently.

This is a spinoff of `nanobrain.eval`, which scored every backbone on every task through one fixed
probe. That was the right shape for a benchmark and the wrong shape for a competition: here we care
about one backbone (sMRI MAE) and five scores, and each task wants a different method. **Nothing
here imports `nanobrain.eval`, and it should stay that way** — this package may be shared with
people who won't get the eval suite.

## Layout

| File | |
|---|---|
| `datasets.py` | core, **frozen**. One `load_fomo_task<k>()` per task, streaming the challenge zips into an HF dataset. Raw niftis, no resampling — the backbone transform does that. |
| `backbone.py` | core, **frozen**. Frozen sMRI MAE encoder; the transform canonicalizes to RAS, rescales to 1mm, fits to the pretraining shape, z-scores in a mean-threshold brain mask. |
| `utils.py` | core. Seeding, git sha, logging. |
| `main_task<k>.py` | shell. One task, end to end. Task 1 is the worked example; copy it. |
| `build.py` + `Apptainer.def` | shell. Package a run dir into the challenge `.sif`. Shared by every task. |

`datasets.py` and `backbone.py` are settled and their caches are warm. Treat them as read-only:
new work goes in `main_task<k>.py`. If one of them genuinely needs to change, that is a
conversation first, because it invalidates every score already recorded.

## The pattern

`main_task1.py` is in three sections, and the split is the point of the whole design.

**`Task1Method` — the part we tune.** Features, head, hyperparameters, anything that might move
the score. Its interface is:

```python
method.fit(rows)              # rows are dataset records: subject, label, images
method.predict(images)        # -> the challenge's output for one subject
method.save(model_dir)        # config.yaml + head.joblib
Task1Method.load(model_dir, **overrides)
```

**The protocol — fixed.** Pool out-of-fold predictions over all subjects, bootstrap subjects for
the CI. No repeats, no stratification; the bootstrap is the only variance estimate. Leave-one-out
where n is tiny (task 1, n=21), **20-fold** where it isn't. Once a task's scheme is set, hold it or
scores stop being comparable across iterations. That is also why `cross_validate` seeds its shuffle
at 0 rather than from `cfg.seed`: the folds are part of the protocol, so tuning the run's seed must
not silently redraw them. At n=48 that seed was worth 0.036 AUROC, so this is not hypothetical.

**Two entrypoints.** `train` runs the protocol then fits a head on all subjects and saves it;
`predict` is the challenge CLI. Both go through `Method.predict`, which is why every fold
exercises the code the submission will run.

That last point is the load-bearing one. `predict` is not a wrapper written at packaging time — it
is the same call cross-validation already made once per held-out subject. When you add a task, keep
that property.

```bash
uv run python -m fomo_tune.main_task1 train modalities=[dwi_b1000,flair] name=task1_dwi_flair
uv run python -m fomo_tune.main_task5 predict --t1 t1.nii.gz --output prob.txt \
    --model-dir experiments/fomo_tune_baseline/output/task5/model
```

`train` takes omegaconf dotlist overrides against the `Config` dataclass at the top of the file,
and writes `config.yaml`, `log.txt`, `metrics.json` and `model/` into `{output_root}/{name}/`.

## Status

Tasks 1, 5 and 3 are complete: verified, packaged, and passing the challenge validator.
**Task 2 is a first draft** — it runs end to end and its geometry is verified, but it has no
recorded run in an experiment dir and is not packaged. **Task 4 is still tabled.**

`experiments/fomo_tune_baseline`, `vitl_fomo300`, one H100:

| run | cross-validated | wall |
|---|---|---|
| `task1`, dwi_b1000, n=21, LOO | AUROC **0.990** [0.944, 1.000] | 11s |
| `task5`, t1w, n=48, 20-fold | AUROC **0.984** [0.953, 1.000] | 68s |
| `task3`, t1w, n=494, 20-fold | r **0.963** [0.957, 0.969], MAE **3.69y** [3.45, 3.95] | 306s |

**Tasks 1 and 3 were submitted to the validation leaderboard on 2026-08-12.** Three attempts
total, so two remain.

**These scores are not yet trusted** — see `.claude/NOTES.md` thread 13. Task 3 in particular is
near published brain-age SOTA off a frozen encoder and a ridge. Subject-level folds and in-fold
hyperparameter selection have been ruled out as the cause; a cohort confound has not. Everything
verified below concerns whether the containers compute what the CV computed, which is a different
question and would not catch a confound.

Task 1's earlier probe sweep got 0.954 [0.861, 1.000] on the same checkpoint
(`experiments/eval_global_0728`), so it roughly reproduces — the gap is LOO vs 5×5 stratified CV,
one interpolation instead of two, and a head selected on AUROC instead of balanced accuracy.

Three checks worth repeating per task, all scripted in `.claude/scratch/` (untracked, so they are
local to this clone):

- `verify_task1.py`, `verify_task35.py <k>` — features are **bit-identical** whether the nifti
  comes from the HF dataset wrapper or from `nib.load` off disk, and the `predict` CLI agrees with
  the in-process method
- `predict_all.py` + `compare_container.py` — the container reproduces the host **bit-identically**
  across all 563 subjects, and the in-sample score is sane. Task 3's in-sample MAE is 2.28y against
  3.69y cross-validated; that ~1.4y of optimism is what a ridge on 1024 features at n=494 should
  cost, and a much smaller gap would have implied the CV was leaking
- `e2e_check.sh` — the packaged container, run the way the challenge runs it, returns the recorded
  value to all six decimals

## What changes per task

| Task | n | Inputs | Output | Split | Notes |
|---|---|---|---|---|---|
| 1 infarct | 21 | adc, dwi_b1000, flair (+t2s/swi) | probability | LOO | done |
| 5 polymicrogyria | 48 | t1w | probability | 20-fold | done |
| 3 brain age | 494 | t1w | age in years | 20-fold | done — RidgeCV head, **Pearson r and MAE**, each with its own bootstrap CI |
| 2 meningioma | 23 | dwi_b1000, flair (+t2s/swi) | mask, input grid | LOO | drafted — flair only, per-subject **Dice** |
| 4 trigeminal | 40 | t2w | mask, labels 1=nerve 2=vessel | — | tabled |

The challenge CLI flag is `--t1` for tasks 3 and 5 both, including task 3, whose file in the zip is
`t1w.nii.gz`.

Task 3 is the only regression, so its `score` drops task 1's guard against bootstrap resamples with
fewer than two distinct labels. The analogous degenerate case is a resample with no spread in `y`,
where Pearson r is undefined rather than merely unstable — at n=494 it does not happen.

Task 4's label order (1=nerve, 2=vessel) is still a guess and needs confirming against the
challenge data before per-class numbers mean anything.

Task 2 is the only one whose `predict` returns a nifti rather than a number, and the only one
needing localized features. It does not use `patch_coords`: `patch_ids` indexes the regular
26x30x26 patch grid directly, so each token's 8x8x8 block is known exactly and nothing has to be
matched by nearest neighbour. Its head predicts each patch's **tumour fraction** — two weighted
rows per patch, `t_i` positives and `512 - t_i` negatives, which is voxel-level logistic
regression collapsed losslessly, so no voxel is ever subsampled. The threshold on that fraction is
selected by an inner 5-fold inside each training fold, never on the training subjects' own
predictions, whose probabilities are over-separated and would put the cut too high.

## Gotchas

**Raw niftis are on disk** at `data/fomo_eval/Task_<k>/preprocessed/<sub>/ses-01/`, the easy way to
exercise `predict` on a real file rather than one written out of the dataset. Task 5 breaks the
naming: `Task_5/preprocessed/sub_01/ses_01/t1.nii.gz` — underscores throughout, and `t1` not `t1w`.
`datasets.py` handles it; anything you write by hand won't.

**`data/fomo_eval` is a symlink** to `/data/smri-datasets/fomo_eval`, so binding the repo into a
container does not bring the data with it. Bind the resolved path too.

**Volumes are wildly anisotropic.** Task 1's DWI is 0.46×0.46×**5.6**mm, so the transform
upsamples z by 5.6× to reach 1mm iso. Nothing is wrong, but don't read the 1mm grid as real
resolution.

**The backbone never saw skull or neck.** Pretraining used a SynthSeg brain mask; the transform
substitutes a mean-intensity threshold, which keeps both. Known fidelity gap — see
`.claude/memory/smri-mae-preprocessing-gap.md`.

**Probabilities are not calibrated.** `LogisticRegressionCV` on ~20 samples × 1024 features shrinks
hard; task 1's out-of-fold probabilities all land in 0.48–0.52 with near-perfect ranking. Fine for
AUROC, which is what the challenge scores, but don't read them as probabilities. Task 5's do span
0–1, which is n=48 rather than n=21 and not evidence of calibration.

**n is tiny, so the CI is the result.** Task 1's is ~0.06 wide at the top of the range. Most tuning
deltas you chase will be inside it. `.claude/NOTES.md` thread 1 has the longer argument.

**GPUs need an allocation** — the login node has no driver. See the `gpu-session` skill.

## Submission

`build.py` packages a run dir into the `.sif` the challenge wants:

```bash
uv run python -m fomo_tune.build experiments/fomo_tune_baseline/output/task1
```

It stages `/app`, then builds from there:

```
/app/predict.py          # shim: calls fomo_tune.main_task<k> predict
/app/model/config.yaml   # from the run dir
/app/model/head.joblib   # from the run dir
/app/model/backbone.pth  # stripped checkpoint, --ckpt-path points here
```

**Both `build.py` and `Apptainer.def` are shared across tasks.** Nothing in staging or in the
dependency list is task-specific; the one thing that varies is the module the shim imports, and
that comes from `task` in the run's saved config — so a run dir knows which task it belongs to.

`predict.py` is **generated at build time** rather than checked in, because its whole meaning is
the container layout staged around it. The logic still lives in `main_task<k>.py`, exercised once
per fold; the shim only picks the subcommand and two paths.

**`Apptainer.def` is not buildable where it sits.** Its `%files` paths are relative to the build
cwd, which is the staging dir. Pointing `apptainer build` at it in the repo fails confusingly; go
through `build.py`.

The run dir deliberately does *not* carry backbone weights — that checkpoint is 3.9G and would be
copied on every run. `--ckpt-path` overrides what `config.yaml` recorded, so the saved config stays
a faithful record of what trained rather than being rewritten at build time. **The staged
checkpoint is stripped to `model` and `args`**, 3.9G → 1.3G, the rest being optimizer state
inference never reads.

**The base image is `python:3.11-slim`, not a CUDA image.** The PyPI torch wheel *is* the cu128
build and vendors the whole CUDA userspace as `nvidia-*` packages, so all the container needs from
the host is the driver, which `--nv`/`--nvccli` binds in. That keeps the SIF at 5.0G against
roughly double for `pytorch/pytorch` and far more for NGC. Versions are pinned to the training
environment mostly so `head.joblib` unpickles against the numpy/sklearn that wrote it.

Apptainer caches the bootstrap layers but **always re-runs `%post`**, so every build re-downloads
~3G of wheels. If that gets annoying, bake a deps-only base SIF and `Bootstrap: localimage` off it.

### Validating

`third_party/container-validator` is the challenge's own validator, test niftis included:

```bash
python third_party/container-validator/container_validator/validate.py \
    --task task1 --sif experiments/fomo_tune_baseline/output/task1/task1.sif
```

It runs `python /app/predict.py --flair /input/… --output /output/<sid>.txt` inside an `apptainer
instance` with `/input`, `/output` and `/tmp` bound — exactly the shim's contract, so nothing in
`predict.py` is guessing at the interface. **All three containers pass**: 20 tests for task 1, 13
for task 5, 12 for task 3. The counts differ because the regression suite drops the
probability-range check, which is worth knowing — nothing the validator runs constrains task 3's
output beyond finiteness, so a badly wrong head would still pass.

One thing easy to miss: it takes GPU via `--nvccli` rather than `--nv`, and one of its tests runs
`nvidia-smi -L` **inside** the container. `python:3.11-slim` ships no `nvidia-smi`, so that test
passes only because `--nvccli` injects the host one — a CUDA base image would hide that dependency
rather than remove it.

**Build on the login node, run under `salloc --nodelist=n-6`.** Those are the only two hosts with
apptainer, and only the latter has a driver — see
`.claude/memory/sif-builds-need-apptainers-apparmor-profile.md`. A driver-less host does not fall
back to CPU; `predict` dies at the forward pass.
