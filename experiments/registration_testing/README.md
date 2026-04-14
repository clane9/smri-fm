# Registration testing

Download testing data:

```bash
uvx hf download medarc/smri-fm --type dataset \
  --include 'openneuro_testing/*' --local-dir ./data
```

Or symlink if you already have it locally

```bash
mkdir data
ln -s PATH_TO_OPENNEURO_TESTING data/openneuro_testing
```

Run the test script

```bash
OFFSET=0 LIMIT=64 VERSION=v1 N_THREADS=1 bash run.sh
OFFSET=0 LIMIT=64 VERSION=v2 N_THREADS=8 bash run.sh
OFFSET=0 LIMIT=64 VERSION=v3 N_THREADS=8 bash run.sh
```
