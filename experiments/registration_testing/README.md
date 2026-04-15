# Registration testing

## Download data

```bash
uvx hf download medarc/smri-fm --type dataset \
  --include 'openneuro_testing/*' --local-dir ./data
```

Or symlink if you already have it locally

```bash
mkdir data
ln -s PATH_TO_OPENNEURO_TESTING data/openneuro_testing
```

## Testing SimpleITK

First trying variants of rigid registration implemented directly with [SimpleITK](https://simpleitk.org/).

```bash
OFFSET=0 LIMIT=64 VERSION=v1 N_THREADS=1 bash run.sh
OFFSET=0 LIMIT=64 VERSION=v2 N_THREADS=8 bash run.sh
OFFSET=0 LIMIT=64 VERSION=v3 N_THREADS=8 bash run.sh
OFFSET=0 LIMIT=64 VERSION=v4 N_THREADS=8 bash run.sh
OFFSET=0 LIMIT=64 VERSION=v5 N_THREADS=8 bash run.sh
OFFSET=0 LIMIT=64 VERSION=v6 N_THREADS=8 bash run.sh
OFFSET=0 LIMIT=64 VERSION=v7 N_THREADS=8 bash run.sh
OFFSET=0 LIMIT=64 VERSION=v8 N_THREADS=8 bash run.sh
OFFSET=0 LIMIT=64 VERSION=v9 N_THREADS=8 bash run.sh
OFFSET=0 LIMIT=64 VERSION=v10 N_THREADS=8 bash run.sh
OFFSET=0 LIMIT=64 VERSION=v11 N_THREADS=8 bash run.sh
```

- v1 copies [BrainIAC](https://github.com/AIM-KannLab/BrainIAC/blob/ba60f45bed832ee0e5683c678caeaeefe2072f0d/src/preprocessing/mri_preprocess_3d_simple.py#L31)
- v2-v11 try out different configs searching for one that produces robust results. I tried a random walk around:
  - gradient descent line search vs fixed lr
  - varying learning rate
  - varying sampling percentage
  - varying shrink factors
  - winsorization
  - brain and whole-head loss masking
  - resampling moving image to isometric 1mm resolution

None of the attempts produced robust results. All had at least one gross failure out of the first 64 test examples.

I also frequently get this error
```
RuntimeError: Exception thrown in SimpleITK ImageRegistrationMethod_Execute: /tmp/SimpleITK-build/ITK-prefix/include/ITK-5.4/itkMattesMutualInformationImageToImageMetricv4.hxx:307:
ITK ERROR: MattesMutualInformationImageToImageMetricv4(0x642b470fc410): All samples map outside moving image buffer. The images do not sufficiently overlap. They need to be initialized to have more overlap before this metric will work. For instance, you can align the image centers by translation
```
which occurs when the optimizer takes a large bad step (see [here](https://github.com/SimpleITK/SimpleITK/issues/1106#issuecomment-628022213)). This error seems especially common for T2w/FLAIR images.

## Testing ANTs

ANTs is the standard tool for brain MRI registration. It is effectively a wrapper around ITK, but with a lot of these kinds of issues sorted out.

I implemented a version `v1_ants`, using [NiWrap](https://niwrap.dev/) to handle interfacing with the ANTs CLI.

```bash
OFFSET=0 LIMIT=64 VERSION=v1_ants N_THREADS=8 bash run.sh
```

It is much more robust, producing good registration on 64/64 testing images.
It is slow though (~20s on 8 threads). And the multiple layers of calling (python -> niwrap -> docker -> ANTs) is bloated. It would be nice to have a config in SimpleITK that just works just as well.
