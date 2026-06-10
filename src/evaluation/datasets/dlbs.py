import io

import fsspec
import pandas as pd
from datasets import Dataset, Features, Value

PROTOCOL = "s3"
STORAGE_OPTIONS = {"anon": True}
ROOT = "openneuro.org/ds004856"

COLUMNS = {
    "AgeMRI_W1": Value("int32"),
    "Sex": Value("string"),
    "EduComp": Value("int32"),
    "BMI_W1": Value("float32"),
    "MMSE_W1": Value("float32"),
}

fs = fsspec.filesystem(PROTOCOL, **STORAGE_OPTIONS)


def create_dlbs():
    paths = sorted(fs.glob(f"{ROOT}/**/*_ses-wave1_acq-MPRAGE_run-1_T1w.nii.gz"))
    participants = pd.read_csv(io.BytesIO(fs.cat_file(f"{ROOT}/participants.tsv")), sep="\t")
    participants = participants.set_index("participant_id")

    features = Features(
        {
            "participant_id": Value("string"),
            **COLUMNS,
            "path": Value("string"),
            "data": Value("binary"),
            "format": Value("string"),
        }
    )

    dataset = Dataset.from_generator(
        _generate_samples,
        features=features,
        gen_kwargs={"paths": paths, "participants": participants, "columns": tuple(COLUMNS)},
    )
    return dataset


def _generate_samples(paths: list[str], participants: pd.DataFrame, columns: list[str]):
    for path in paths:
        sub = path.split("/")[2]
        row = participants.loc[sub, :]
        name = path.split("/")[-1]
        stem, ext = name.split(".", maxsplit=1)
        data = fs.cat_file(path)
        record = {
            "participant_id": sub,
            **{col: row[col] for col in columns},
            "path": path.removeprefix(ROOT).lstrip("/"),
            "data": data,
            "format": ext,
        }
        print(path, sub, f"{len(data) / 1e6:.1f}MB")
        yield record


if __name__ == "__main__":
    ds = create_dlbs()
    print(ds)
