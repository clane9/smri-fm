import gzip
import io
import importlib.resources

import fsspec
import pandas as pd
import nibabel as nib
from datasets import Dataset, Features, Value, Nifti

ROOT = "openneuro.org/ds004856"


def create_dlbs_t1w():
    paths = get_dlbs_t1w_filelist()

    columns = {
        "AgeMRI_W1": Value("int32"),
        "Sex": Value("string"),
        "EduComp": Value("int32"),
        "BMI_W1": Value("float32"),
        "MMSE_W1": Value("float32"),
    }

    dataset = Dataset.from_generator(
        _generate_dlbs_t1w_samples,
        features=Features(
            {
                "participant_id": Value("string"),
                **columns,
                "path": Value("string"),
                "nifti": Nifti(),
            }
        ),
        gen_kwargs={"paths": paths, "columns": tuple(columns)},
        num_proc=8,
    )
    return dataset


def _generate_dlbs_t1w_samples(paths: list[str], columns: tuple[str]):
    fs = fsspec.filesystem("s3", anon=True)

    participants = pd.read_csv(
        io.BytesIO(fs.cat_file(f"{ROOT}/participants.tsv")),
        sep="\t",
        dtype_backend="pyarrow",
    )
    participants = participants.set_index("participant_id")

    for path in paths:
        sub = path.split("/")[0]
        row = participants.loc[sub, :].to_dict()
        buf = fs.cat_file(f"{ROOT}/{path}")
        img = load_nifti_bytes(buf)

        record = {
            "participant_id": sub,
            **{col: row[col] for col in columns},
            "path": path,
            "nifti": img,
        }
        yield record


def get_dlbs_t1w_filelist() -> list[str]:
    files = importlib.resources.files("evaluation.datasets.dlbs")
    with files.joinpath("dlbs_wave1_t1w_images.txt").open() as f:
        paths = f.read().strip().splitlines()
    return paths


def load_nifti_bytes(buf: bytes):
    # gzip magic number, see https://stackoverflow.com/a/76055284/9534390 or "Magic number" on https://en.wikipedia.org/wiki/Gzip
    if buf[:2] == b"\x1f\x8b":
        buf = gzip.decompress(buf)
    img = nib.Nifti1Image.from_bytes(buf)
    return img


if __name__ == "__main__":
    ds = create_dlbs_t1w()
    print(ds)
