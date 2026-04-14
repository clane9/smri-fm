import argparse
import json

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


VERSIONS = {"v1": rigid_registration_v1}


def rigid_registration_cli():
    parser = argparse.ArgumentParser(prog="rigid_registration")
    parser.add_argument("input", help="path to input image")
    parser.add_argument("output", help="path for registered output")
    parser.add_argument("--version", choices=list(VERSIONS), default="v1")
    args = parser.parse_args()

    template_path = tflow.get(
        "MNI152NLin6Asym", desc=None, resolution=1, suffix="T1w", extension="nii.gz"
    )
    moving_img = sitk.ReadImage(args.input, sitk.sitkFloat32)
    fixed_img = sitk.ReadImage(str(template_path), sitk.sitkFloat32)

    result_img, transform, info = VERSIONS[args.version](moving_img, fixed_img)
    sitk.WriteImage(result_img, args.output)
    print(json.dumps({"input": args.input, "version": args.version, **info}))
