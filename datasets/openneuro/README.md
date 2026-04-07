# OpenNeuro

link: https://openneuro.org/

## Download

Download all raw anatomical images, JSON sidecars, and participant tables from the official OpenNeuro S3 bucket.

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ data/openneuro \
  --exclude '*' \
  --include '*T1w.*' \
  --include '*T2w.*' \
  --include '*FLAIR.*' \
  --include '*participants.*' \
  --exclude '*bidsignore*' \
  --exclude '*derivatives*' \
  --exclude '*desc-preproc*'
```

The data are also backed up to the MedARC R2 bucket. Download from the backup with.

```bash
aws s3 sync s3://medarc/smri-datasets/source/openneuro data/openneuro
```

## Index

We have pre-computed indexes of the images and participants:

- [`metadata/openneuro_images.csv`](metadata/openneuro_images.csv): image metadata (e.g. modality, shape, resolution, dtype)
- [`metadata/openneuro_participants.csv`](metadata/openneuro_participants.csv): subject table with age and sex

To re-compute the indexes, run
```bash
uv run scripts/index_images.py
uv run scripts/index_participants.py
```
