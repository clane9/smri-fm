import argparse
import json
import time
from functools import partial
from typing import Literal

import numpy as np
import SimpleITK as sitk
import templateflow.api as tflow


def to_iso(img: sitk.Image, resolution: float = 1.0) -> sitk.Image:
    size = img.GetSize()
    spacing = img.GetSpacing()
    if all(s == resolution for s in spacing):
        return img
    new_spacing = 3 * (resolution,)
    new_size = [int(round(size[ii] * spacing[ii] / resolution)) for ii in range(3)]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputDirection(img.GetDirection())
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(0.0)
    resample.SetOutputPixelType(sitk.sitkFloat32)
    return resample.Execute(img)


def rigid_registration(
    moving_img: sitk.Image,
    fixed_img: sitk.Image,
    *,
    n_histogram_bins: int = 50,
    sampling_strategy: Literal["random", "regular"] = "random",
    sampling_percentage: float = 0.01,
    optimizer: Literal["gradient_descent", "gradient_descent_line_search"] = "gradient_descent",
    learning_rate: float = 1.0,
    max_iterations: int = 100,
    shrink_factors: list[int] = (4, 2, 1),
    smoothing_sigmas: list[int] = (2, 1, 0),
    winsorize: bool = False,
    winsorize_range: tuple[float, float] = (0.5, 99.5),
    fixed_mask: sitk.Image | None = None,
    moving_to_iso: bool = False,
    max_step_size: float | None = None,
):
    temp_img = moving_img
    if moving_to_iso:
        temp_img = to_iso(temp_img)
    if winsorize:
        arr = sitk.GetArrayFromImage(temp_img)
        lo, hi = np.percentile(arr[arr > 0], list(winsorize_range))
        temp_img = sitk.Clamp(temp_img, temp_img.GetPixelIDValue(), float(lo), float(hi))

    transform = sitk.CenteredTransformInitializer(
        fixed_img,
        temp_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=n_histogram_bins)
    registration_method.SetMetricSamplingStrategy(
        registration_method.REGULAR
        if sampling_strategy == "regular"
        else registration_method.RANDOM
    )
    registration_method.SetMetricSamplingPercentage(sampling_percentage)
    registration_method.SetInterpolator(sitk.sitkLinear)

    optimizer_kwargs = dict(
        learningRate=learning_rate,
        numberOfIterations=max_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    if max_step_size:
        optimizer_kwargs["maximumStepSizeInPhysicalUnits"] = max_step_size
    if optimizer == "gradient_descent_line_search":
        registration_method.SetOptimizerAsGradientDescentLineSearch(**optimizer_kwargs)
    else:
        registration_method.SetOptimizerAsGradientDescent(**optimizer_kwargs)

    if fixed_mask is not None:
        registration_method.SetMetricFixedMask(fixed_mask)

    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=list(shrink_factors))
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=list(smoothing_sigmas))
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(transform)

    final_transform = registration_method.Execute(fixed_img, temp_img)

    result_img = sitk.Resample(
        moving_img, fixed_img, final_transform, sitk.sitkBSpline, 0.0, moving_img.GetPixelID()
    )

    info = {
        "stop_cond": registration_method.GetOptimizerStopConditionDescription(),
        "final_metric": registration_method.GetMetricValue(),
    }
    return result_img, final_transform, info


VERSIONS = {
    "v1": rigid_registration,
    "v2": partial(
        rigid_registration, sampling_percentage=0.1, optimizer="gradient_descent_line_search"
    ),
    "v3": partial(
        rigid_registration,
        n_histogram_bins=32,
        sampling_strategy="regular",
        sampling_percentage=0.25,
        learning_rate=0.1,
        max_iterations=1000,
        shrink_factors=[8, 4, 2, 1],
        smoothing_sigmas=[3, 2, 1, 0],
        winsorize=True,
    ),
    "v4": rigid_registration,  # also uses fixed brain mask
    "v5": partial(rigid_registration, sampling_percentage=0.1),
    "v6": partial(
        rigid_registration,
        n_histogram_bins=32,
        sampling_percentage=0.25,
        shrink_factors=[8, 4, 2, 1],
        smoothing_sigmas=[3, 2, 1, 0],
    ),
    "v7": partial(rigid_registration, sampling_percentage=0.1, moving_to_iso=True),
    "v8": partial(
        rigid_registration,
        sampling_percentage=0.25,
        optimizer="gradient_descent_line_search",
        max_iterations=1000,
        shrink_factors=(4,),
        smoothing_sigmas=(2,),
        moving_to_iso=True,
    ),
    "v9": partial(
        rigid_registration,
        sampling_percentage=0.25,
        optimizer="gradient_descent_line_search",
        max_iterations=1000,
        shrink_factors=(4,),
        smoothing_sigmas=(2,),
        moving_to_iso=True,
    ),  # same as v8 but with head mask
    "v10": partial(
        rigid_registration,
        sampling_strategy="regular",
        sampling_percentage=0.25,
        optimizer="gradient_descent_line_search",
        max_iterations=1000,
        shrink_factors=(4,),
        smoothing_sigmas=(2,),
        moving_to_iso=True,
    ),
    "v11": partial(
        rigid_registration,
        sampling_percentage=0.25,
        optimizer="gradient_descent_line_search",
        learning_rate=0.1,
        max_iterations=1000,
        shrink_factors=(4,),
        smoothing_sigmas=(2,),
        moving_to_iso=True,
        max_step_size=4.0,
    ),
}


def rigid_registration_cli():
    parser = argparse.ArgumentParser(prog="rigid_registration")
    parser.add_argument("input", help="path to input image")
    parser.add_argument("output", help="path for registered output")
    parser.add_argument("--n-threads", "-j", type=int, default=1)
    parser.add_argument("--version", choices=list(VERSIONS), default="v1")
    args = parser.parse_args()

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.n_threads)

    template_path = tflow.get(
        "MNI152NLin6Asym", desc=None, resolution=1, suffix="T1w", extension="nii.gz"
    )
    moving_img = sitk.ReadImage(args.input, sitk.sitkFloat32)
    fixed_img = sitk.ReadImage(str(template_path), sitk.sitkFloat32)

    if args.version == "v4":
        mask_path = tflow.get(
            "MNI152NLin6Asym", desc="brain", resolution=1, suffix="mask", extension="nii.gz"
        )
        fixed_mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    elif args.version == "v9":
        fixed_mask = sitk.Cast(fixed_img > 600, sitk.sitkUInt8)
    else:
        fixed_mask = None

    t0 = time.perf_counter()
    result_img, transform, info = VERSIONS[args.version](
        moving_img, fixed_img, fixed_mask=fixed_mask
    )
    sitk.WriteImage(result_img, args.output)
    elapsed = time.perf_counter() - t0
    record = {"input": args.input, "version": args.version, **info, "run_time": elapsed}
    print(json.dumps(record))
