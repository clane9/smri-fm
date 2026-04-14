import argparse
import json
import time

import numpy as np
import SimpleITK as sitk
import templateflow.api as tflow


def rigid_registration_v1(moving_img: sitk.Image, fixed_img: sitk.Image):
    # Initialize transform
    transform = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Set up registration method
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(transform)

    # Execute registration
    final_transform = registration_method.Execute(fixed_img, moving_img)

    # Apply transform
    moving_img = sitk.Resample(
        moving_img, fixed_img, final_transform, sitk.sitkBSpline, 0.0, fixed_img.GetPixelID()
    )

    stop_condition = registration_method.GetOptimizerStopConditionDescription()
    final_metric = registration_method.GetMetricValue()
    info = {"stop_cond": stop_condition, "final_metric": final_metric}

    return moving_img, final_transform, info


def rigid_registration_v2(moving_img: sitk.Image, fixed_img: sitk.Image):
    # Initialize transform
    transform = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Set up registration method
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(transform)

    # Execute registration
    final_transform = registration_method.Execute(fixed_img, moving_img)

    # Apply transform
    moving_img = sitk.Resample(
        moving_img, fixed_img, final_transform, sitk.sitkBSpline, 0.0, fixed_img.GetPixelID()
    )

    stop_condition = registration_method.GetOptimizerStopConditionDescription()
    final_metric = registration_method.GetMetricValue()
    info = {"stop_cond": stop_condition, "final_metric": final_metric}

    return moving_img, final_transform, info


def rigid_registration_v3(moving_img: sitk.Image, fixed_img: sitk.Image):
    # Winsorize moving image intensities for metric computation (ANTs [0.005, 0.995])
    arr = sitk.GetArrayFromImage(moving_img)
    lo, hi = np.percentile(arr[arr > 0], [0.5, 99.5])
    moving_winsorized = sitk.Clamp(moving_img, moving_img.GetPixelIDValue(), float(lo), float(hi))

    # Initialize transform
    transform = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_winsorized,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Set up registration method
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.25)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.1,
        numberOfIterations=1000,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(transform)

    # Execute registration on winsorized image
    final_transform = registration_method.Execute(fixed_img, moving_winsorized)

    # Apply transform to original (non-winsorized) image
    moving_img = sitk.Resample(
        moving_img, fixed_img, final_transform, sitk.sitkBSpline, 0.0, fixed_img.GetPixelID()
    )

    stop_condition = registration_method.GetOptimizerStopConditionDescription()
    final_metric = registration_method.GetMetricValue()
    info = {"stop_cond": stop_condition, "final_metric": final_metric}

    return moving_img, final_transform, info


VERSIONS = {"v1": rigid_registration_v1, "v2": rigid_registration_v2, "v3": rigid_registration_v3}


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

    t0 = time.perf_counter()
    result_img, transform, info = VERSIONS[args.version](moving_img, fixed_img)
    sitk.WriteImage(result_img, args.output)
    elapsed = time.perf_counter() - t0
    record = {"input": args.input, "version": args.version, **info, "run_time": elapsed}
    print(json.dumps(record))
