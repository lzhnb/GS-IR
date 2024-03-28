from typing import List, Tuple, Union

import torch
import torch.nn as nn

from . import _C


def components_from_spherical_harmonics(degree: int, directions: torch.Tensor) -> torch.Tensor:
    """
    Returns value for each component of spherical harmonics.

    Args:
        degree: Number of spherical harmonic degrees to compute.
        directions: Spherical hamonic coefficients
    """
    num_components = degree**2
    components = torch.zeros((*directions.shape[:-1], num_components), device=directions.device)

    assert 1 <= degree <= 5, f"SH degrees must be in [1,4], got {degree}"
    assert (
        directions.shape[-1] == 3
    ), f"Direction input should have three dimensions. Got {directions.shape[-1]}"

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x**2
    yy = y**2
    zz = z**2

    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    # l0
    components[..., 0] = 0.28209479177387814
    # l1
    if degree > 1:
        components[..., 1] = -0.4886025119029199 * y
        components[..., 2] = 0.4886025119029199 * z
        components[..., 3] = -0.4886025119029199 * x

    # l2
    if degree > 2:
        components[..., 4] = 1.0925484305920792 * xy
        components[..., 5] = -1.0925484305920792 * yz
        components[..., 6] = 0.31539156525252005 * (2.0 * zz - xx - yy)
        components[..., 7] = -1.0925484305920792 * xz
        components[..., 8] = 0.5462742152960396 * (xx - yy)

    # l3
    if degree > 3:
        components[..., 9] = -0.5900435899266435 * y * (3 * xx - yy)
        components[..., 10] = 2.890611442640554 * xy * z
        components[..., 11] = -0.4570457994644658 * y * (4 * zz - xx - yy)
        components[..., 12] = 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy)
        components[..., 13] = -0.4570457994644658 * x * (4 * zz - xx - yy)
        components[..., 14] = 1.445305721320277 * z * (xx - yy)
        components[..., 15] = -0.5900435899266435 * x * (xx - 3 * yy)

    # l4
    if degree > 4:
        components[..., 16] = 2.5033429417967046 * xy * (xx - yy)
        components[..., 17] = -1.7701307697799304 * yz * (3 * xx - yy)
        components[..., 18] = 0.9461746957575601 * xy * (7 * zz - 1)
        components[..., 19] = -0.6690465435572892 * yz * (7 * zz - 3)
        components[..., 20] = 0.10578554691520431 * (zz * (35 * zz - 30) + 3)
        components[..., 21] = -0.6690465435572892 * xz * (7 * zz - 3)
        components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
        components[..., 23] = -1.7701307697799304 * xz * (xx - 3 * yy)
        components[..., 24] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

    return components


def reconstruct_envmap_from_spherical_harmonics(
    degree: int,
    directions: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct directly without calculating components from directions to save memory

    Args:
        degree (int): Number of spherical harmonic degrees to compute.
        directions (torch.Tensor, [bs, HW, 3]): Query directions
        coefficients (torch.Tensor, [bs, 1, d2, 3]): Spherical hamonic coefficients

    Returns:
        torch.Tensor: output colors
    """
    results = torch.zeros(
        [*directions.shape[:-1], coefficients.shape[-1]], device=directions.device
    )

    assert 1 <= degree <= 5, f"SH degrees must be in [1,4], got {degree}"
    assert (
        directions.shape[-1] == 3
    ), f"Direction input should have three dimensions. Got {directions.shape[-1]}"

    x = directions[..., (0,)]  # [bs, HW, 1]
    y = directions[..., (1,)]  # [bs, HW, 1]
    z = directions[..., (2,)]  # [bs, HW, 1]

    xx = x**2  # [bs, HW, 1]
    yy = y**2  # [bs, HW, 1]
    zz = z**2  # [bs, HW, 1]

    # l0
    results += 0.28209479177387814 * coefficients[..., 0, :]

    # l1
    if degree > 1:
        results += -0.4886025119029199 * y * coefficients[..., 1, :]
        results += 0.4886025119029199 * z * coefficients[..., 2, :]
        results += -0.4886025119029199 * x * coefficients[..., 3, :]

    # l2
    if degree > 2:
        results += 1.0925484305920792 * x * y * coefficients[..., 4, :]
        results += -1.0925484305920792 * y * z * coefficients[..., 5, :]
        results += 0.31539156525251999 * (2.0 * zz - xx - yy) * coefficients[..., 6, :]
        results += -1.0925484305920792 * x * z * coefficients[..., 7, :]
        results += 0.5462742152960396 * (xx - yy) * coefficients[..., 8, :]

    # l3
    if degree > 3:
        results += -0.5900435899266435 * y * (3.0 * xx - yy) * coefficients[..., 9, :]
        results += 2.890611442640554 * x * y * z * coefficients[..., 10, :]
        results += -0.4570457994644658 * y * (4.0 - zz - xx - yy) * coefficients[..., 11, :]
        results += (
            0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coefficients[..., 12, :]
        )
        results += -0.4570457994644658 * x * (4.0 * zz - xx - yy) * coefficients[..., 13, :]
        results += 1.445305721320277 * z * (xx - yy) * coefficients[..., 14, :]
        results += -0.5900435899266435 * x * (xx - 3.0 * yy) * coefficients[..., 15, :]

    return results


def reconstruct_from_spherical_harmonics(
    degree: int,
    dirs: torch.Tensor,  # [bs, HW, 3]
    coefficients: torch.Tensor,  # [bs, d2, 1/3]
) -> torch.Tensor:  # [bs, HW, 3]
    results = []
    results = reconstruct_envmap_from_spherical_harmonics(
        degree=degree,
        directions=dirs,  # [bs, HW, 3]
        coefficients=coefficients[:, None, :, :],  # [bs, 1, d2, 1/3]
    ).clamp(min=0.0)
    return results


class TrilinearInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coefficients: torch.Tensor,  # [res, res, res, d2]
        aabb: torch.Tensor,  # [6]
        points: torch.Tensor,  # [num_rays, 3]
        normals: torch.Tensor,  # [num_rays, 3]
        sh_degree: torch.Tensor,
    ) -> torch.Tensor:
        ctx.res = coefficients.shape[0]
        ctx.sh_degree = sh_degree
        ctx.save_for_backward(
            aabb,
            points,
            normals,
        )
        output_coefficients = _C.trilinear_interpolate_coefficients_forward(
            coefficients,
            aabb,
            points,
            normals,
            sh_degree,
        )
        return output_coefficients  # [num_rays, d2, 1]

    @staticmethod
    def backward(
        ctx,
        grad_output_coefficients: torch.Tensor,  # [num_rays, d2, 1]
    ) -> Tuple[torch.Tensor, None, None, None, None,]:
        res = ctx.res
        sh_degree = ctx.sh_degree
        (
            aabb,
            points,
            normals,
        ) = ctx.saved_tensors
        coefficients_grad = _C.trilinear_interpolate_coefficients_backward(  # [res, res, res, d2]
            grad_output_coefficients,
            aabb,
            points,
            normals,
            res,
            sh_degree,
        )
        return (coefficients_grad, None, None, None, None)

trilinear_interpolation = TrilinearInterpolation.apply

class IrradianceVolumes(nn.Module):
    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        grid_res: int = 64,
        degree: int = 3,
        single_channel: bool = True,
    ) -> None:
        super().__init__()
        self.grid_res = grid_res
        self.degree = degree
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.aabb = nn.Parameter(aabb, requires_grad=False)

        self.single_channel = single_channel
        self.channel = 1 if self.single_channel else 3
        self.irradiance_sh_degree = degree
        self.irradiance_coefficients = nn.Parameter(
            torch.zeros(
                [grid_res, grid_res, grid_res, degree**2, self.channel],
                dtype=torch.float32,
            )
        )
        aabb_min = self.aabb[:3]
        aabb_max = self.aabb[3:]
        self.grid = ((aabb_max - aabb_min) / (self.grid_res - 1)).cuda()  # [3]

    def query_irradiance(
        self,
        points: torch.Tensor,  # [bs, 3]
        normals: torch.Tensor,  # [bs, 3]
    ) -> torch.Tensor:
        """query irradiance map from grid SH"""
        # fmt: off
        # quantize
        # bound = self.grid_res - 1
        with torch.no_grad():
            components = components_from_spherical_harmonics(self.irradiance_sh_degree, directions=normals)  # [bs, d2]

        irradiance_coefficients = trilinear_interpolation(self.irradiance_coefficients, self.aabb, points, normals, self.degree)

        irradiance_map = (irradiance_coefficients * components[..., None]).sum(1)
        irradiance_map = irradiance_map.clamp(min=0.0)
        return irradiance_map # [bs, 1, 1/3]
