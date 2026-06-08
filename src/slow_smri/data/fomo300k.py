import zipfile
import gzip
from pathlib import Path

import fsspec
import nibabel as nib
import numpy as np
from torch.utils.data import IterableDataset

from slow_smri.data.prefetch import prefetch

DEFAULT_SUFFIXES = ("T1w", "T2w", "FLAIR")


class Fomo300K(IterableDataset):
    def __init__(
        self,
        root: str,
        pattern: str | list[str] = "**/*.zip",
        suffix: str | list[str] = DEFAULT_SUFFIXES,
        filelist: str | None = None,
        shuffle: bool = False,
        random_state: int | np.random.Generator | None = None,
        cache_dir: str | Path | None = None,
        max_workers: int = 1,
        storage_options: dict | None = None,
    ):
        self.root = root
        self.pattern = pattern
        self.suffix = suffix
        self.filelist = filelist
        self.shuffle = shuffle
        self.random_state = random_state
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.storage_options = storage_options

        print(f"loading FOMO300K: {self.root}")
        fs, root_ = fsspec.url_to_fs(str(root), **(storage_options or {}))

        if filelist is not None:
            print(f"loading filelist: {filelist}")
            with fsspec.open(filelist, "rt") as f:
                paths = f.read().strip().splitlines()
        else:
            patterns = [pattern] if isinstance(pattern, str) else list(pattern)
            print(f"finding files matching: {patterns}")
            paths = sorted(
                p.removeprefix(root_).lstrip("/")
                for pat in patterns
                for p in fs.glob(f"{root_}/{pat}")
            )
        examples = "\n".join(map(str, paths[:5]))
        print(f"found {len(paths)} files:\n{examples}\n...")

        self.paths_ = np.array(paths)
        self.rng_ = np.random.default_rng(random_state)

    def __iter__(self):
        paths = self.paths_.copy()
        if self.shuffle:
            self.rng_.shuffle(paths)

        for path, fullpath in prefetch(
            self.root,
            paths,
            cache_dir=self.cache_dir,
            max_workers=self.max_workers,
            storage_options=self.storage_options,
        ):
            for name, img in read_fomo300_zip(fullpath, suffix=self.suffix):
                fullname = f"{path.removesuffix('.zip')}/{name}"
                yield fullname, img


def read_fomo300_zip(path: str, suffix: str | list[str] = DEFAULT_SUFFIXES):
    suffixes = {suffix} if isinstance(suffix, str) else set(suffix)

    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".nii.gz"):
                continue
            stem = name.removesuffix(".nii.gz")
            suf = stem.split("_")[-1]
            if suf not in suffixes:
                continue

            with z.open(name) as zf:
                with gzip.open(zf) as gz:
                    img = nib.Nifti1Image.from_stream(gz)
                    data = img.get_fdata(dtype=np.float32)
                    affine = img.affine

            img = nib.Nifti1Image(data, affine)
            yield name, img


if __name__ == "__main__":
    import time
    import importlib.resources

    ds = Fomo300K(
        "hf://datasets/FOMO-MRI/FOMO300K",
        filelist=importlib.resources.files("slow_smri.config").joinpath(
            "fomo300k_full_filelist.txt"
        ),
        max_workers=8,
        shuffle=True,
    )

    t0 = time.time()
    total_mb = 0.0
    for name, img in ds:
        shape = img.shape
        data = img.get_fdata(dtype=np.float32)
        numel = (data > data.mean()).sum()
        total_mb = total_mb + numel / 1e6
        total_mb_s = total_mb / (time.time() - t0)

        orient = "".join(nib.aff2axcodes(img.affine))
        spacing = tuple(round(float(s), 3) for s in img.header.get_zooms())
        print(f"{name} {shape} {orient} {spacing} {numel / 1000:.0f}K {total_mb_s:.0f} MB/s")
