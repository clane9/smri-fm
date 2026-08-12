"""FOMO task 2: meningioma segmentation, scored by per-subject Dice as the challenge scores it.

`Task2Method` is the part we tune -- features, head, hyperparameters. The protocol below it is
fixed so scores stay comparable across iterations: leave one subject out, per-subject Dice on the
subject's own grid, bootstrap subjects for the CI.

The head predicts, for each patch, the fraction of that patch's voxels that are tumour. Two
weighted rows per patch -- `t_i` positives and `512 - t_i` negatives -- is voxel-level logistic
regression over every voxel of every patch, collapsed losslessly, so nothing is subsampled.

`train` runs that protocol then fits and saves a head; `predict` is the challenge contract,
modality paths in and a mask nifti out. Both go through `Task2Method.predict`, so every fold
exercises the path the submission will run.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import torch
from einops import reduce, repeat
from omegaconf import OmegaConf
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from fomo_tune.backbone import load_backbone
from fomo_tune.utils import git_sha, set_seed, setup_logging

logger = logging.getLogger("fomo_tune")

Images = dict[str, nib.Nifti1Image]

# Probabilities sit near the 2.4e-4 voxel prevalence, so the grid is geometric rather than linear.
THRESHOLDS = np.logspace(-6, -0.3, 60)


@dataclass
class Config:
    task: str = "task2"
    ckpt_path: str = (
        "/data/mihir-stuff/smri-pretrained/pretrain_full_90_10_h100/checkpoint-last.pth"
    )
    modality: str = "flair"
    output_root: str = "output/fomo_tune"
    name: str = "task2"
    inverse_reg: float = 1.0
    n_inner: int = 5
    largest_component: bool = True
    device: str = "cuda"
    seed: int = 4466


# ---- geometry -----------------------------------------------------------------------------


def repack(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Round-trip through nibabel: the HF Nifti wrapper's own reorientation is not trustworthy."""
    return nib.Nifti1Image(img.dataobj, img.affine, img.header)


def resample_nearest(
    volume: np.ndarray, source_affine: np.ndarray, target_affine: np.ndarray, target_shape: tuple
) -> np.ndarray:
    """`volume` read at every voxel of the target grid, nearest neighbour, zero outside it.

    General over affines rather than assuming the transform's axis-aligned steps, so a volume that
    is not already RAS costs accuracy nowhere and never silently transposes.
    """
    transform = np.linalg.inv(source_affine) @ target_affine
    grid = np.indices(target_shape, dtype=np.float64)
    index = np.tensordot(transform[:3, :3], grid, axes=1) + transform[:3, 3].reshape(3, 1, 1, 1)
    index = np.rint(index).astype(np.int64)

    inside = np.all((index >= 0) & (index < np.array(volume.shape).reshape(3, 1, 1, 1)), axis=0)
    flat = np.ravel_multi_index(np.where(inside, index, 0), volume.shape)
    return np.where(inside, volume.reshape(-1)[flat], 0)


# ---- method: the part we tune ---------------------------------------------------------------


def fit_head(features: np.ndarray, counts: np.ndarray, voxels: int, inverse_reg: float) -> Pipeline:
    """Logistic regression on the voxels of every patch, collapsed to two weighted rows each.

    Unbalanced on purpose: the fitted probability is then the patch's tumour fraction, which is
    the quantity the threshold cuts. Balancing would buy conditioning and cost that meaning.
    """
    positive = counts > 0
    x = np.concatenate([features, features[positive]])
    y = np.concatenate([np.zeros(len(features)), np.ones(positive.sum())])
    weight = np.concatenate([voxels - counts, counts[positive]])

    keep = weight > 0
    clf = LogisticRegression(C=inverse_reg, max_iter=1000)
    head = make_pipeline(StandardScaler(), clf)
    head.fit(x[keep], y[keep], logisticregression__sample_weight=weight[keep])
    return head


def selected_patches(
    probabilities: np.ndarray, threshold: float, grid_size: tuple, largest_component: bool
) -> np.ndarray:
    """Patches above the threshold, optionally reduced to their largest connected blob."""
    selected = probabilities >= threshold
    if not largest_component or not selected.any():
        return selected
    labels, n_components = ndimage.label(selected.reshape(grid_size))
    if n_components == 1:
        return selected
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    return (labels == sizes.argmax()).reshape(-1)


def patch_dice(selected: np.ndarray, counts: np.ndarray, voxels: int) -> float:
    """Dice of a patch selection against the labels, on the backbone's own 1mm grid.

    Predictions are constant within a patch, so `2 * sum(t_i for i in S) / (|S| * 512 + T)` is the
    voxel Dice exactly, without building a volume. Used to choose the threshold, not to score.
    """
    total = counts.sum()
    denominator = selected.sum() * voxels + total
    return float(2 * counts[selected].sum() / denominator) if denominator else 1.0


class Task2Method:
    """Frozen sMRI MAE, per-patch tokens, logistic head on the patch tumour fraction."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backbone, self.transform = load_backbone(cfg.ckpt_path)
        self.device = torch.device(cfg.device)
        self.backbone.to(self.device).eval().requires_grad_(False)
        self.modality = cfg.modality

        patchify = self.backbone.encoder.patchify
        self.grid_size = tuple(patchify.grid_size)
        self.patch_size = tuple(patchify.patch_size)
        self.img_size = tuple(patchify.img_size)
        self.voxels_per_patch = int(np.prod(self.patch_size))

        self.cache: dict[str, tuple] = {}
        self.head = None
        self.threshold = None

    @torch.inference_mode()
    def embed(self, images: Images) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Patch features (N, D), their indices into the flattened patch grid, and the affine of
        the grid they live on. A pure function of the images, so training and inference agree."""
        sample = self.transform(images[self.modality])
        batch = {key: value[None].to(self.device) for key, value in sample.items()}

        with torch.autocast("cuda", torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.backbone(batch)

        keep = out["token_mask"][0].bool()
        features = out["patch_embeds"][0][keep].float().cpu().numpy()
        patch_ids = out["patch_ids"][0][keep].cpu().numpy()
        return features, patch_ids, sample["affine"].numpy()

    def patch_counts(self, seg: nib.Nifti1Image, affine: np.ndarray) -> np.ndarray:
        """Tumour voxels per patch, over the whole grid, in the encoder's flattened order."""
        seg = repack(seg)
        labels = np.asarray(seg.dataobj, dtype=np.float32).round()
        on_grid = resample_nearest(labels, seg.affine, affine, self.img_size)
        px, py, pz = self.patch_size
        return reduce(on_grid, "(gx px) (gy py) (gz pz) -> (gx gy gz)", "sum", px=px, py=py, pz=pz)

    def probabilities(self, images: Images) -> tuple[np.ndarray, np.ndarray]:
        """Tumour fraction per patch over the whole grid -- zero where the encoder kept no token,
        which is a patch with no in-brain voxel -- and the affine of that grid."""
        features, patch_ids, affine = self.embed(images)
        grid = np.zeros(int(np.prod(self.grid_size)))
        grid[patch_ids] = self.head.predict_proba(features)[:, self.positive]
        return grid, affine

    def training_patches(self, rows: list[dict]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per subject, the kept patches' features, their grid indices, and the whole grid's
        tumour-voxel counts. Cached: leave-one-out revisits every subject n times."""
        for row in rows:
            if row["subject"] not in self.cache:
                features, patch_ids, affine = self.embed(row)
                counts = self.patch_counts(row["seg"], affine)
                self.cache[row["subject"]] = (features, patch_ids, counts)
        return [self.cache[row["subject"]] for row in rows]

    def choose_threshold(self, subjects: list[tuple], n_inner: int) -> float:
        """The cut maximizing mean per-subject Dice on out-of-fold predictions from an inner CV.

        In-sample probabilities are over-separated, so a threshold picked on the training
        subjects' own predictions sits too high and under-segments everything else.
        """
        n_patches = int(np.prod(self.grid_size))
        dice = np.zeros((len(subjects), len(THRESHOLDS)))
        folds = KFold(n_splits=n_inner, shuffle=True, random_state=0)
        for train, test in folds.split(subjects):
            head = fit_head(
                np.concatenate([subjects[i][0] for i in train]),
                np.concatenate([subjects[i][2][subjects[i][1]] for i in train]),
                self.voxels_per_patch,
                self.cfg.inverse_reg,
            )
            positive = list(head[-1].classes_).index(1)
            for i in test:
                features, patch_ids, counts = subjects[i]
                grid = np.zeros(n_patches)
                grid[patch_ids] = head.predict_proba(features)[:, positive]
                dice[i] = [
                    patch_dice(
                        selected_patches(
                            grid, threshold, self.grid_size, self.cfg.largest_component
                        ),
                        counts,
                        self.voxels_per_patch,
                    )
                    for threshold in THRESHOLDS
                ]
        return float(THRESHOLDS[dice.mean(axis=0).argmax()])

    def fit(self, rows: list[dict]) -> None:
        subjects = self.training_patches(rows)
        self.threshold = self.choose_threshold(subjects, self.cfg.n_inner)
        self.head = fit_head(
            np.concatenate([features for features, _, _ in subjects]),
            np.concatenate([counts[patch_ids] for _, patch_ids, counts in subjects]),
            self.voxels_per_patch,
            self.cfg.inverse_reg,
        )
        self.positive = list(self.head[-1].classes_).index(1)

    def predict(self, images: Images) -> nib.Nifti1Image:
        """A binary mask on the input's own grid, which is what the challenge scores."""
        grid, affine = self.probabilities(images)
        selected = selected_patches(
            grid, self.threshold, self.grid_size, self.cfg.largest_component
        )
        px, py, pz = self.patch_size
        gx, gy, gz = self.grid_size
        volume = repeat(
            selected.astype(np.uint8),
            "(gx gy gz) -> (gx px) (gy py) (gz pz)",
            gx=gx,
            gy=gy,
            gz=gz,
            px=px,
            py=py,
            pz=pz,
        )

        image = repack(images[self.modality])
        mask = resample_nearest(volume, affine, image.affine, image.shape)
        return nib.Nifti1Image(mask.astype(np.uint8), image.affine)

    def save(self, model_dir: Path) -> None:
        """Everything `load` needs but the backbone weights, which stay wherever `ckpt_path`
        points -- a few hundred KB, so a run saves one without copying a 3.7G checkpoint."""
        model_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, model_dir / "config.yaml")
        state = {"head": self.head, "positive": self.positive, "threshold": self.threshold}
        joblib.dump(state, model_dir / "head.joblib")

    @classmethod
    def load(cls, model_dir: Path, **overrides) -> "Task2Method":
        """Rebuild a fitted method from `save`. Overrides are Config fields, for what differs
        between here and the container -- the backbone path, the device."""
        cfg = OmegaConf.merge(
            OmegaConf.structured(Config), OmegaConf.load(model_dir / "config.yaml"), overrides
        )
        method = cls(cfg)
        state = joblib.load(model_dir / "head.joblib")
        method.head = state["head"]
        method.positive = state["positive"]
        method.threshold = state["threshold"]
        return method


# ---- protocol: the part we hold fixed ---------------------------------------------------

# Every image the task ships. The method picks which of them it wants, as at inference, where the
# challenge hands over all the modalities whether or not a model uses them.
IMAGE_COLS = ("dwi_b1000", "flair")


def dice_score(prediction: np.ndarray, truth: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(truth.sum())
    return 2 * int(np.logical_and(prediction, truth).sum()) / denominator if denominator else 1.0


def leave_one_out(rows: list[dict], method: Task2Method) -> list[dict]:
    """Out-of-fold mask for every subject, each predicted by a head fit on the other n-1.

    Alongside the score, two diagnostics: the same subject's Dice at the best threshold anyone
    could have picked, which bounds what threshold selection can win, and the patch-grid Dice,
    which should track the native one if the geometry is right.
    """
    records = []
    start = time.perf_counter()
    for row in rows:
        method.fit([r for r in rows if r["subject"] != row["subject"]])
        images = {key: row[key] for key in IMAGE_COLS}

        prediction = np.asarray(method.predict(images).dataobj) > 0
        truth = np.asarray(repack(row["seg"]).dataobj).round() > 0
        assert prediction.shape == truth.shape, "prediction is not on the label grid"

        grid, affine = method.probabilities(images)
        counts = method.patch_counts(row["seg"], affine)
        oracle = max(
            patch_dice(
                selected_patches(grid, threshold, method.grid_size, method.cfg.largest_component),
                counts,
                method.voxels_per_patch,
            )
            for threshold in THRESHOLDS
        )
        chosen = selected_patches(
            grid, method.threshold, method.grid_size, method.cfg.largest_component
        )

        record = {
            "subject": row["subject"],
            "dice": dice_score(prediction, truth),
            "dice_patch": patch_dice(chosen, counts, method.voxels_per_patch),
            "dice_oracle": oracle,
            "threshold": method.threshold,
            "predicted_voxels": int(prediction.sum()),
            "true_voxels": int(truth.sum()),
        }
        records.append(record)
        logger.info(
            f"fold {len(records)}/{len(rows)} {row['subject']} dice={record['dice']:.3f} "
            f"(patch {record['dice_patch']:.3f}, oracle {oracle:.3f}) "
            f"thr={method.threshold:.2e} vox={record['predicted_voxels']}/{record['true_voxels']} "
            f"({time.perf_counter() - start:.0f}s)"
        )
    return records


def score(records: list[dict], seed: int = 0, n_boot: int = 2000, alpha: float = 0.05) -> dict:
    """Mean per-subject Dice, the challenge metric, plus a percentile CI resampling subjects."""
    dice = np.array([record["dice"] for record in records])
    rng = np.random.default_rng(seed)
    samples = dice[rng.integers(0, len(dice), size=(n_boot, len(dice)))].mean(axis=1)
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "dice": float(dice.mean()),
        "dice_ci_low": float(low),
        "dice_ci_high": float(high),
        "dice_patch": float(np.mean([record["dice_patch"] for record in records])),
        "dice_oracle": float(np.mean([record["dice_oracle"] for record in records])),
    }


# ---- entrypoints ------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    # imported here, not at the top, so the container needs no dataset stack to run `predict`
    from fomo_tune.datasets import load_fomo_task2

    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.from_dotlist(args.overrides))
    run_dir = Path(cfg.output_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)
    set_seed(cfg.seed)
    logger.info(f"run {cfg.name} (git {git_sha()})")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg).rstrip()}")
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # decoded once: leave-one-out revisits every subject n times, and the niftis are small
    rows = list(load_fomo_task2())
    logger.info(f"dataset: {len(rows)} subjects")

    method = Task2Method(cfg)
    start = time.perf_counter()
    records = leave_one_out(rows, method)
    run_time = time.perf_counter() - start
    summary = score(records)

    # the shipped head sees all n subjects, so it is not any of the models scored above
    method.fit(rows)
    method.save(run_dir / "model")

    record = {"name": cfg.name, **summary, "run_time": round(run_time, 1)}
    (run_dir / "metrics.json").write_text(json.dumps(record) + "\n")
    (run_dir / "per_subject.json").write_text(json.dumps(records, indent=2) + "\n")
    scores = "  ".join(f"{k}={v:.4f}" for k, v in summary.items())
    logger.info(f"result: {scores}  ({run_time:.0f}s)")


def predict(args: argparse.Namespace) -> None:
    """The challenge contract: modality paths in, a mask nifti written to `--output`.

    `/app/predict.py` in the container is a shim over this, so what scores the submission is the
    code leave-one-out already ran, not something generated at build time.
    """
    overrides = {"device": args.device}
    if args.ckpt_path:
        overrides["ckpt_path"] = args.ckpt_path
    method = Task2Method.load(args.model_dir, **overrides)

    # every image the challenge hands over, as in `leave_one_out`; the method takes what it uses
    paths = {"dwi_b1000": args.dwi, "flair": args.flair}
    mask = method.predict({key: nib.load(path) for key, path in paths.items()})

    nib.save(mask, args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(required=True)

    train_parser = modes.add_parser("train", help="leave-one-out over the task, then fit and save")
    train_parser.add_argument("overrides", nargs="*", help="config overrides, e.g. device=cpu")
    train_parser.set_defaults(run=train)

    predict_parser = modes.add_parser("predict", help="one subject, one mask nifti")
    for flag in ("--flair", "--dwi"):
        predict_parser.add_argument(flag, type=Path, required=True)
    # accepted and ignored: the 4th modality is t2s on some subjects and swi on others
    for flag in ("--t2s", "--swi"):
        predict_parser.add_argument(flag, type=Path)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--model-dir", type=Path, default=Path("/app/model"))
    predict_parser.add_argument("--ckpt-path", help="overrides the trained config's backbone path")
    predict_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    predict_parser.set_defaults(run=predict)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
