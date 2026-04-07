from pathlib import Path

import pandas as pd

SEX_MAP = {"m": "M", "male": "M", "man": "M", "f": "F", "female": "F", "woman": "F"}


def normalize_sub(val: str):
    try:
        return val.removeprefix("sub-")
    except AttributeError:
        return pd.NA


def normalize_sex(val):
    try:
        return SEX_MAP.get(str(val).strip().lower(), pd.NA)
    except (ValueError, TypeError):
        return pd.NA


def normalize_age(val):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return pd.NA


def load_participants(path: Path, dataset: str) -> pd.DataFrame | None:
    df = pd.read_csv(path, sep="\t", dtype=str)
    cols_lower = {c.lower(): c for c in df.columns}

    if "participant_id" not in df:
        return None

    sex_col = cols_lower.get("sex") or cols_lower.get("gender")
    age_col = cols_lower.get("age")

    out = pd.DataFrame()
    out["dataset"] = len(df) * [dataset]
    out["sub"] = df["participant_id"].map(normalize_sub)
    out["sex"] = df[sex_col].map(normalize_sex) if sex_col else pd.NA
    out["age"] = df[age_col].map(normalize_age) if age_col else pd.NA
    out = out.dropna(subset=["sub"])
    return out


def main():
    root = Path("data/openneuro")
    outpath = Path("metadata/openneuro_participants.csv")

    frames = []
    for tsv in sorted(root.glob("*/participants.tsv")):
        dataset = tsv.parent.name
        try:
            result = load_participants(tsv, dataset)
            if result is not None:
                frames.append(result)
        except Exception as e:
            print(f"SKIP {tsv}: {e}")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outpath, index=False)
    print(f"Saved {len(df)} rows to {outpath}")
    print(df[["sub", "sex", "age"]].count())


if __name__ == "__main__":
    main()
