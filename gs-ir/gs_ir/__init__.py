import torch

from . import _C
from .volumes import IrradianceVolumes

@torch.no_grad()
def recon_occlusion(
    H: int,
    W: int,
    bound: float,
    points: torch.Tensor,  # [HW, 3]
    normals: torch.Tensor,  # [HW, 3]
    occlusion_coefficients: torch.Tensor,
    occlusion_ids: torch.Tensor,
    aabb: torch.Tensor,
    sample_rays: int = 256,
    degree: int = 4,
) -> torch.Tensor:
    occlu_res = occlusion_ids.shape[0]
    half_grid = bound / float(occlu_res)
    shift_points = points + normals * half_grid
    # shift_points = points
    (
        coefficients,  # [HW, d2, 1]
        coeff_ids,  # [HW, 8]
    ) = _C.sparse_interpolate_coefficients(
        occlusion_coefficients,
        occlusion_ids,
        aabb,
        shift_points,
        normals,
        degree,
    )
    coefficients = coefficients.permute(0, 2, 1)  # [HW, 1, d2]

    roughness = torch.ones([H * W, 1], dtype=torch.float32).cuda()
    occlusion = _C.SH_reconstruction(
        coefficients, normals, roughness, sample_rays, degree
    )  # [HW, 1]

    return occlusion


__all__ = ["_C", "recon_occlusion", "IrradianceVolumes"]
