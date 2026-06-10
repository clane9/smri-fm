from pathlib import Path

import fsspec

ROOT = "openneuro.org/ds004856"
OUT_ROOT = Path(__file__).parent


def main():
    outpath = OUT_ROOT / "dlbs_wave1_t1w_images.txt"
    if outpath.exists():
        print(f"output {outpath} exists; exiting")
        return

    fs = fsspec.filesystem("s3", anon=True)
    paths: list[str] = sorted(fs.glob(f"{ROOT}/**/*_ses-wave1_acq-MPRAGE_run-1_T1w.nii.gz"))
    paths = [p.removeprefix(ROOT).lstrip("/") for p in paths]
    examples = "\n".join(paths[:5])
    print(f"found {len(paths)} images:\n{examples}\n...")

    with outpath.open("w") as f:
        print("\n".join(paths), file=f)


if __name__ == "__main__":
    main()
