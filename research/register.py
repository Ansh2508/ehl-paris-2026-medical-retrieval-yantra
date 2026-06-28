"""Lightweight rigid registration to a canonical reference (SimpleITK, Mattes MI).

Tests finding ②: L2's damage is *independent rigid rotation* of query vs target.
If we rigidly register every volume into ONE common reference frame, that rotation
is undone and MIND/embeddings become viable on L2 again. Cross-modal-safe (MI metric),
no training. This is the cheap proof-of-concept; EasyReg/SynthMorph is the robust upgrade.
"""
from __future__ import annotations
import torch


def _to_sitk(vol: torch.Tensor):
    import SimpleITK as sitk
    arr = vol.squeeze(0).detach().cpu().float().numpy()      # [D,H,W]
    return sitk.Cast(sitk.GetImageFromArray(arr), sitk.sitkFloat32)


def _to_tensor(img) -> torch.Tensor:
    import SimpleITK as sitk
    return torch.from_numpy(sitk.GetArrayFromImage(img)).unsqueeze(0).float()


def _linear_stage(R, f, m, transform, iters):
    import SimpleITK as sitk
    init = sitk.CenteredTransformInitializer(
        f, m, transform, sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)   # cross-modal-safe
    R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=iters)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1]); R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SetInitialTransform(init, inPlace=False)
    return R.Execute(f, m)


def register_to_ref(moving: torch.Tensor, fixed: torch.Tensor, mode: str = "rigid",
                    iters: int = 80) -> torch.Tensor:
    """Register `moving` [1,R,R,R] onto `fixed`'s frame; return resampled moving.
    mode: 'rigid' (6-DOF, undoes rotation), 'affine' (12-DOF, +scale/shear),
          'deformable' (affine init + BSpline free-form -> undoes the L2 elastic warp).
    Cross-modal-safe (Mattes MI), no training."""
    import SimpleITK as sitk
    f, m = _to_sitk(fixed), _to_sitk(moving)
    try:
        if mode == "rigid":
            t = _linear_stage(sitk.ImageRegistrationMethod(), f, m, sitk.Euler3DTransform(), iters)
        else:
            # affine first (also the init for deformable)
            t = _linear_stage(sitk.ImageRegistrationMethod(), f, m, sitk.AffineTransform(3), iters)
            if mode == "deformable":
                R2 = sitk.ImageRegistrationMethod()
                R2.SetMovingInitialTransform(t)                       # compose on top of affine
                bspline = sitk.BSplineTransformInitializer(f, [5, 5, 5])
                R2.SetInitialTransform(bspline, inPlace=True)
                R2.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
                R2.SetMetricSamplingStrategy(R2.RANDOM); R2.SetMetricSamplingPercentage(0.1)
                R2.SetInterpolator(sitk.sitkLinear)
                R2.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=60)
                R2.SetShrinkFactorsPerLevel([4, 2]); R2.SetSmoothingSigmasPerLevel([2, 1])
                tb = R2.Execute(f, m)
                t = sitk.CompositeTransform([t, tb])
    except Exception:
        t = sitk.CenteredTransformInitializer(
            f, m, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    out = sitk.Resample(m, f, t, sitk.sitkLinear, 0.0, m.GetPixelID())
    return _to_tensor(out)
