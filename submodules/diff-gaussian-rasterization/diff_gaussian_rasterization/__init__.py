#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple, Optional, Tuple

import kornia
import torch.nn as nn
import torch

from . import _C


def cpu_deep_copy_tuple(input_tuple: Tuple) -> Tuple:
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple
    ]
    return tuple(copied_tensors)


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    inference: bool
    argmax_depth: bool


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        opacities: torch.Tensor,
        normal: torch.Tensor,
        albedo: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
        sh: torch.Tensor,
        colors_precomp: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov3Ds_precomp: torch.Tensor,
        raster_settings: GaussianRasterizationSettings,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            normal,
            albedo,
            roughness,
            metallic,
            scales,
            rotations,
            cov3Ds_precomp,
            sh,
            raster_settings.campos,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.scale_modifier,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.sh_degree,
            raster_settings.prefiltered,
            raster_settings.argmax_depth,
            raster_settings.inference,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                (
                    num_rendered,
                    color,
                    radii,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                    opacity_map,
                    depth,
                    out_normal,
                    albedo_map,
                    roughness_map,
                    metallic_map,
                ) = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                color,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                opacity_map,
                depth,
                out_normal,
                albedo_map,
                roughness_map,
                metallic_map,
            ) = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            normal,
            albedo,
            roughness,
            metallic,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        return (
            color,
            radii,
            opacity_map,
            depth,
            out_normal,
            albedo_map,
            roughness_map,
            metallic_map,
        )

    @staticmethod
    def backward(
        ctx,
        grad_out_color: torch.Tensor,
        gard_radii: Optional[torch.Tensor] = None,
        grad_out_opacity: Optional[torch.Tensor] = None,
        grad_depth: Optional[torch.Tensor] = None,
        grad_out_normal: Optional[torch.Tensor] = None,
        grad_out_albedo: Optional[torch.Tensor] = None,
        grad_out_roughness: Optional[torch.Tensor] = None,
        grad_out_metallic: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
    ]:
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            normal,
            albedo,
            roughness,
            metallic,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            normal,
            albedo,
            roughness,
            metallic,
            scales,
            rotations,
            cov3Ds_precomp,
            sh,
            raster_settings.campos,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.scale_modifier,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.sh_degree,
            grad_out_color,
            grad_out_opacity,
            grad_out_normal,
            grad_out_albedo,
            grad_out_roughness,
            grad_out_metallic,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            num_rendered,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_normal,
                    grad_albedo,
                    grad_roughness,
                    grad_metallic,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_normal,
                grad_albedo,
                grad_roughness,
                grad_metallic,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_rotations,
            ) = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_opacities,
            grad_normal,
            grad_albedo,
            grad_roughness,
            grad_metallic,
            grad_sh,
            grad_colors_precomp,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions: torch.Tensor) -> torch.Tensor:
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions, raster_settings.viewmatrix, raster_settings.projmatrix
            )

        return visible

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        opacities: torch.Tensor,
        normal: torch.Tensor,
        albedo: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
        shs: Optional[torch.Tensor] = None,
        colors_precomp: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        cov3D_precomp: Optional[torch.Tensor] = None,
        derive_normal: bool = False,

    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception("Please provide excatly one of either SHs or precomputed colors!")

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        (
            color,
            radii,
            opacity_map,
            depth,
            out_normal,
            albedo_map,
            roughness_map,
            metallic_map,
        ) = _RasterizeGaussians.apply(
            means3D,
            means2D,
            opacities,
            normal,
            albedo,
            roughness,
            metallic,
            shs,
            colors_precomp,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )

        if derive_normal:
            focal_x = raster_settings.image_width / (2.0 * raster_settings.tanfovx)
            focal_y = raster_settings.image_height / (2.0 * raster_settings.tanfovy)
            # # NOTE: trick to smooth depth for better normal
            # depth_filter = depth
            depth_filter = kornia.filters.median_blur(depth[None, ...], (3, 3))[0]
            normal_from_depth = _C.depth_to_normal(
                raster_settings.image_width,
                raster_settings.image_height,
                focal_x,
                focal_y,
                raster_settings.viewmatrix,
                depth_filter,
            )
        else:
            normal_from_depth = torch.zeros_like(out_normal)

        return (
            color,
            radii,
            opacity_map,
            depth,
            normal_from_depth,
            out_normal,
            albedo_map,
            roughness_map,
            metallic_map,
        )
