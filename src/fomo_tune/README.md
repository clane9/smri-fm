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
| `backbone.py` | core, **frozen**. `load_backbone(ckpt_path) -> (SmriMaeBackbone, SmriMaeTransform)`. Frozen sMRI MAE encoder; the transform canonicalizes to RAS, rescales to 1mm, fits to the pretraining shape, z-scores in a mean-threshold brain mask. |
| `utils.py` | core. `set_seed`, `git_sha`, `setup_logging`. |
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
the CI. No repeats, no stratification; the bootstrap is the only variance estimate. Splitting is
per-task but fixed within a task — leave-one-out where n is tiny (task 1, n=21), **20-fold** where
it isn't (tasks 3 and 5), which is close enough to LOO without paying for 494 refits. Once a task's
scheme is set, hold it or scores stop being comparable across iterations.

**Two entrypoints.** `train` runs the protocol then fits a head on all subjects and saves it;
`predict` is the challenge CLI. Both go through `Method.predict`, which is why every fold
exercises the code the submission will run.

That last point is the load-bearing one. `predict` is not a wrapper written at packaging time — it
is the same call cross-validation already made once per held-out subject. When you add a task, keep
that property.

```bash
uv run python -m fomo_tune.main_task1 train modalities=[dwi_b1000,flair] name=task1_dwi_flair
uv run python -m fomo_tune.main_task1 predict --model-dir output/fomo_tune/task1_dwi/model \
    --adc adc.nii.gz --dwi dwi.nii.gz --flair flair.nii.gz --output prob.txt
```

`train` takes omegaconf dotlist overrides against the `Config` dataclass at the top of the file.
It writes `config.yaml`, `log.txt`, `metrics.json`, and `model/` into `{output_root}/{name}/`.

## Status

Task 1 is drafted, verified, and packaged — its container passes the challenge validator. Tasks 3
and 5 are next. **Tasks 2 and 4 are tabled** — both are
segmentation, both need `predict` to emit a nifti on the input grid, and neither is worth opening
until the classification and regression tasks are settled.

`vitl_fomo300`, dwi_b1000 only, n=21: **AUROC 0.990 [0.944, 1.000]**, about 25s wall on one H100.
The earlier probe sweep got 0.954 [0.861, 1.000] on the same checkpoint
(`experiments/eval_global_0728`), so this roughly reproduces — the gap is LOO vs 5×5 stratified CV,
one interpolation instead of two, and a head selected on AUROC instead of balanced accuracy.

Two checks worth repeating per task (`.claude/scratch/verify_task1.py` did both, though the
on-disk niftis below make the first one easier than it was there):
- features are **bit-identical** whether the nifti comes from the HF dataset wrapper or from
  `nib.load` off disk, so CV numbers transfer to the container
- the `predict` CLI agrees with the in-process method

## What changes per task

Counts and modalities, read from the local zips:

| Task | n | Inputs | Output | Split | Notes |
|---|---|---|---|---|---|
| 1 infarct | 21 | adc, dwi_b1000, flair (+t2s/swi) | probability | LOO | done |
| 5 polymicrogyria | 48 | t1w | probability | 20-fold | **do this one next** |
| 3 brain age | 494 | t1w | age in years | 20-fold | regression: RidgeCV head, scored by **Pearson r and MAE** — `score` returns both, each with its own bootstrap CI |
| 2 meningioma | 23 | dwi_b1000, flair (+t2s/swi) | mask, input grid | — | tabled |
| 4 trigeminal | 40 | t2w | mask, labels 1=nerve 2=vessel | — | tabled |

Task 5 is the cheapest next step: same protocol shape, same head, one modality, AUROC again, and
the only real change is the split.

Task 3 is the first regression, so the head and the metrics both change. Two things task 1's
`score` won't hand you: it returns a single metric, and it guards the bootstrap by skipping
resamples with fewer than two distinct labels. For regression, drop that guard — the analogous
degenerate case is a resample with no spread in `y`, where Pearson r is undefined rather than
merely unstable.

When tasks 2 and 4 come back: `predict` must write a nifti on the input's grid, and the method
needs localized features rather than a pooled vector — `backbone.forward` returns `patch_coords`
in world mm for exactly that. Task 4's label order (1=nerve, 2=vessel) is still a guess and needs
confirming against the challenge data before per-class numbers mean anything.

## Gotchas

**Raw niftis are on disk** at `data/fomo_eval/Task_<k>/preprocessed/<sub>/ses-01/`, which is the
easy way to exercise `predict` on a real file rather than one written out of the dataset:

```bash
uv run python -m fomo_tune.main_task1 predict \
    --model-dir output/fomo_tune/task1_dwi/model \
    --adc  data/fomo_eval/Task_1/preprocessed/sub-20/ses-01/adc.nii.gz \
    --dwi  data/fomo_eval/Task_1/preprocessed/sub-20/ses-01/dwi_b1000.nii.gz \
    --flair data/fomo_eval/Task_1/preprocessed/sub-20/ses-01/flair.nii.gz \
    --output /tmp/prob.txt
```

Task 5 breaks the naming: `Task_5/preprocessed/sub_01/ses_01/t1.nii.gz` — underscores throughout,
and `t1` not `t1w`. `datasets.py` already handles it; anything you write by hand won't.

**Volumes are wildly anisotropic.** Task 1's DWI is 0.46×0.46×**5.6**mm, so the transform
upsamples z by 5.6× to reach 1mm iso. Nothing is wrong, but don't read the 1mm grid as real
resolution.

**The backbone never saw skull or neck.** Pretraining used a SynthSeg brain mask; the transform
substitutes a mean-intensity threshold, which keeps both. Known fidelity gap — see
`.claude/memory/smri-mae-preprocessing-gap.md`.

**Probabilities are not calibrated.** `LogisticRegressionCV` on ~20 samples × 1024 features shrinks
hard; task 1's out-of-fold probabilities all land in 0.48–0.52 with near-perfect ranking. Fine for
AUROC, which is what the challenge scores, but don't read them as probabilities.

**n is tiny, so the CI is the result.** Task 1's is ~0.06 wide at the top of the range. Most tuning
deltas you chase will be inside it. `.claude/NOTES.md` thread 1 has the longer argument.

**GPUs need an allocation** — the login node has no driver. See the `gpu-session` skill.

## Submission

`build.py` packages a run dir into the `.sif` the challenge wants. One command, taking the run dir
the shipped head was saved into:

```bash
uv run python -m fomo_tune.build output/fomo_tune/task1_dwi
```

It stages `/app`, then builds from there:

```
/app/predict.py          # shim: calls fomo_tune.main_task<k> predict
/app/model/config.yaml   # from the run dir
/app/model/head.joblib   # from the run dir
/app/model/backbone.pth  # stripped checkpoint, --ckpt-path points here
```

**Both `build.py` and `Apptainer.def` are shared across tasks**, which they can be because nothing
in staging or in the dependency list is task-specific. The one thing that does vary is the module
the shim imports, and that comes from `task` in the run's saved config — so a run dir knows which
task it belongs to, and `build.py` never needs telling.

`predict.py` is **generated at build time** rather than checked in. It is eight lines whose whole
meaning is the container layout staged around it, so there is nowhere outside a container to run
it. This does not weaken the point above about `predict` not being written at packaging time: the
logic still lives in `main_task<k>.py`, exercised once per fold, and the shim only picks the
subcommand and two paths.

**`Apptainer.def` is not buildable where it sits.** Its `%files` paths are relative to the build
cwd, which is the staging dir. Pointing `apptainer build` at it in the repo fails confusingly; go
through `build.py`.

The run dir deliberately does *not* carry backbone weights — that checkpoint is 3.9G and would be
copied on every run. `--ckpt-path` overrides what `config.yaml` recorded, so the saved config stays
a faithful record of what trained rather than being rewritten at build time.

**The staged checkpoint is stripped to `model` and `args`**, which is 3.9G → 1.3G because the rest
is optimizer state inference never reads. `load_backbone` needs no change for this, and `predict`
gives a bit-identical probability either way (0.524739 on `sub-20`, checked on GPU).

**The base image is `python:3.11-slim`, not a CUDA image.** The PyPI torch wheel *is* the cu128
build and vendors the whole CUDA userspace as `nvidia-*` packages, so all the container needs from
the host is the driver, which `--nv`/`--nvccli` binds in. That keeps the SIF at 5.3G (4.0G of
image, 1.3G of weights) against roughly double for `pytorch/pytorch` and far more for NGC.
Versions are pinned to the training environment
mostly so `head.joblib` unpickles against the numpy/sklearn that wrote it.

Apptainer caches the bootstrap layers but **always re-runs `%post`**, so every build re-downloads
~3G of wheels. If that gets annoying, bake a deps-only base SIF and `Bootstrap: localimage` off it.

### Validating

`third_party/container-validator` is the challenge's own validator, test niftis included:

```bash
python third_party/container-validator/container_validator/validate.py \
    --task task1 --sif output/fomo_tune/task1_dwi/task1.sif
```

It runs `python /app/predict.py --flair /input/… --adc … --dwi … --swi … --output /output/<sid>.txt`
inside an `apptainer instance` with `/input`, `/output` and `/tmp` bound — which is exactly the
shim's contract, so nothing in `predict.py` is guessing at the interface.

One thing it does that is easy to miss: it takes GPU via `--nvccli` rather than `--nv`, and one of
its tests runs `nvidia-smi -L` **inside** the container. `python:3.11-slim` ships no `nvidia-smi`,
so that test passes only because `--nvccli` injects the host one — a CUDA base image would hide
that dependency rather than remove it.

**The `task1_dwi` container passes all 20 validator tests**, and `predict` inside it returns
0.524739 on `sub-20`, identical to the same call outside the container. So the packaging is
verified end to end, not just built.

**Run it on a compute node with apptainer, which as of 2026-08-11 means `n-6`** — `salloc
--nodelist=n-6`. The other nodes fail the validator's preflight. The login node has apptainer but
no driver, and
`predict` there dies inside `can_use_cudnn_attention` — the jagged-SDPA path reaches into CUDA even
when the tensors are on CPU, so a driver-less host fails at the forward pass rather than falling
back. That is the CPU gap worth remembering; it is not a container problem.
