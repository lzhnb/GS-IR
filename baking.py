import os
import itertools
import math
from argparse import ArgumentParser
from os import makedirs
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from tqdm import trange
from diff_gaussian_rasterization import _C
from gs_ir import _C as gs_ir_ext

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getProjectionMatrix
from utils.sh_utils import components_from_spherical_harmonics


def getWorld2ViewTorch(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    Rt = torch.zeros((4, 4), device=R.device)
    Rt[:3, :3] = R[:3, :3].T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt

# inverse the mapping from https://github.com/NVlabs/nvdiffrec/blob/dad3249af8ede96c7dd72c30328272117fabb710/render/light.py#L22
def get_envmap_dirs(res: List[int] = [256, 512]) -> Tuple[torch.Tensor, torch.Tensor]:
    gy, gx = torch.meshgrid(
        torch.linspace(0.0, 1.0 - 1.0 / res[0], res[0], device="cuda"),
        torch.linspace(-1.0, 1.0 - 1.0 / res[1], res[1], device="cuda"),
        indexing="ij",
    )
    d_theta, d_phi = np.pi / res[0], 2 * np.pi / res[1]

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]

    # get solid angles
    solid_angles = ((costheta - torch.cos(gy * np.pi + d_theta)) * d_phi)[..., None]  # [H, W, 1]
    print(f"solid_angles_sum error: {solid_angles.sum() - 4 * np.pi}")

    return solid_angles, reflvec


def lookAt(eye: torch.Tensor, center: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    Z = F.normalize(eye - center, dim=0)
    Y = up
    X = F.normalize(torch.cross(Y, Z), dim=0)
    Y = F.normalize(torch.cross(Z, X), dim=0)

    matrix = torch.tensor(
        [
            [X[0], Y[0], Z[0]],
            [X[1], Y[1], Z[1]],
            [X[2], Y[2], Z[2]],
        ]
    )

    return matrix


def get_canonical_rays(H: int, W: int, tan_fovx: float, tan_fovy: float) -> torch.Tensor:
    cen_x = W / 2
    cen_y = H / 2
    focal_x = W / (2.0 * tan_fovx)
    focal_y = H / (2.0 * tan_fovy)

    x, y = torch.meshgrid(
        torch.arange(W),
        torch.arange(H),
        indexing="xy",
    )
    x = x.flatten()  # [H * W]
    y = y.flatten()  # [H * W]
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cen_x + 0.5) / focal_x,
                (y - cen_y + 0.5) / focal_y,
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [H * W, 3]
    # NOTE: it is not normalized
    return camera_dirs.cuda()


MIN_DEPTH = 1e-6

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--bound", default=1.5, type=float, help="The bound of occlusion volumes.")
    parser.add_argument("--valid", default=1.5, type=float, help="Identify valid area (cull invalid 3D Gaussians) to accelerate baking.")
    parser.add_argument("--occlu_res", default=160, type=int, help="The resolution of the baked occlusion volumes.")
    parser.add_argument("--cubemap_res", default=256, type=int, help="The resolution of the cubemap produced during baking.")
    parser.add_argument("--occlusion", default=0.4, type=float, help="The occlusion threshold to control visible area, the smaller the bound, the lighter the ambient occlusion.")
    parser.add_argument("--checkpoint", type=str, default=None, help="The path to the checkpoint to load.")
    args = get_combined_args(parser)

    model_path = os.path.dirname(args.checkpoint)
    print("Rendering " + model_path)

    dataset = model.extract(args)
    pipeline = pipeline.extract(args)
    gaussians = GaussianModel(4)

    checkpoint = torch.load(args.checkpoint)
    if isinstance(checkpoint, Tuple):
        model_params = checkpoint[0]
    elif isinstance(checkpoint, Dict):
        model_params = checkpoint["gaussians"]
    else:
        raise TypeError
    gaussians.restore(model_params)

    # Set up rasterization configuration
    res = args.cubemap_res
    bg_color = torch.zeros([3, res, res], device="cuda")
    # # NOTE: for debuging HDRi
    bg_colors = [
        torch.zeros([3, res, res], device="cuda"),  # black
        torch.zeros([3, res, res], device="cuda"),  # red
        torch.zeros([3, res, res], device="cuda"),  # green
        torch.zeros([3, res, res], device="cuda"),  # blue
        torch.zeros([3, res, res], device="cuda"),  # yellow
        torch.ones([3, res, res], device="cuda"),  # white
    ]
    # 1-red
    bg_colors[1][0, ...] = 1
    # 2-green
    bg_colors[2][1, ...] = 1
    # 3-blue
    bg_colors[3][2, ...] = 1
    # 4-yellow
    bg_colors[4][:2, ...] = 1

    # NOTE: capture 6 views with fov=90
    rotations: List[torch.Tensor] = [
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, -1.0, 0.0]), torch.tensor([0.0, 0.0, -1.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 0.0, -1.0]), torch.tensor([0.0, 1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
    ]

    zfar = 100.0
    znear = 0.01
    projection_matrix = (
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=math.pi * 0.5, fovY=math.pi * 0.5)
        .transpose(0, 1)
        .cuda()
    )

    # positions = torch.ones([1, 3]).cuda()
    prods = list(itertools.product(range(args.occlu_res), range(args.occlu_res), range(args.occlu_res)))
    aabb_min = torch.tensor([-args.bound] * 3).cuda()
    aabb_max = torch.tensor([args.bound] * 3).cuda()

    grid = (aabb_max - aabb_min) / (args.occlu_res - 1)
    positions = torch.tensor(prods).cuda() * grid + aabb_min  # [bs, 3]

    # init occlusion volume
    occlu_sh_degree = 4
    occlusion_threshold = args.occlusion
    valid_mask = torch.zeros([args.occlu_res, args.occlu_res, args.occlu_res]).bool().cuda()
    points = gaussians.get_xyz
    quat = ((points - aabb_min) // grid).long()
    qx0, qx1 = quat[..., 0].clamp(min=0, max=args.occlu_res - 1), (quat[..., 0] + 1).clamp(
        min=0, max=args.occlu_res - 1
    )
    qy0, qy1 = quat[..., 1].clamp(min=0, max=args.occlu_res - 1), (quat[..., 1] + 1).clamp(
        min=0, max=args.occlu_res - 1
    )
    qz0, qz1 = quat[..., 2].clamp(min=0, max=args.occlu_res - 1), (quat[..., 2] + 1).clamp(
        min=0, max=args.occlu_res - 1
    )
    valid_mask[qx0, qy0, qz0] = True
    valid_mask[qx0, qy0, qz1] = True
    valid_mask[qx0, qy1, qz0] = True
    valid_mask[qx0, qy1, qz1] = True
    valid_mask[qx1, qy0, qz0] = True
    valid_mask[qx1, qy0, qz1] = True
    valid_mask[qx1, qy1, qz0] = True
    valid_mask[qx1, qy1, qz1] = True
    xyz_ids = torch.where(valid_mask)
    num_grid = valid_mask.sum()
    occlusion_ids = (
        torch.ones(
            [args.occlu_res, args.occlu_res, args.occlu_res],
            dtype=torch.int32,
        )
        * -1
    ).cuda()
    occlusion_ids[xyz_ids[0].tolist(), xyz_ids[1].tolist(), xyz_ids[2].tolist()] = torch.arange(
        num_grid, dtype=torch.int32
    ).cuda()
    occlusion_coefficients = torch.zeros(
        [num_grid, occlu_sh_degree**2, 1], dtype=torch.float32
    ).cuda()

    render_path = os.path.join(model_path, "temp")

    makedirs(render_path, exist_ok=True)

    # prepare
    screenspace_points = (
        torch.zeros_like(
            gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=False, device="cuda"
        )
        + 0
    )
    means3D = gaussians.get_xyz
    means2D = screenspace_points
    opacity = gaussians.get_opacity
    shs = gaussians.get_features
    scales = gaussians.get_scaling
    rots = gaussians.get_rotation

    (
        solid_angles,  # [H, W, 1]
        envmap_dirs,  # [H, W, 3]
    ) = get_envmap_dirs()
    components = components_from_spherical_harmonics(occlu_sh_degree, envmap_dirs)  # [H, W, d2]

    # get canonical ray and its norm to normalize depth
    canonical_rays = get_canonical_rays(H=res, W=res, tan_fovx=1.0, tan_fovy=1.0)  # [HW, 3]
    norm = torch.norm(canonical_rays, p=2, dim=-1).reshape(res, res, 1)  # [H, W]

    with torch.no_grad():
        for grid_id in trange(num_grid):
            quat = torch.cat(torch.where(occlusion_ids == grid_id))
            position = positions[(quat[0] * args.occlu_res**2 + quat[1] * args.occlu_res + quat[2],)]
            # position = torch.tensor([0.0, 1.5, 0.0]).to(position.device)
            rgb_cubemap = []
            opacity_cubemap = []
            depth_cubemap = []
            # NOTE: crop by position
            diff = means3D - position
            valid = (diff.abs() < args.valid).all(dim=1)
            valid_means3D = means3D[valid]
            valid_means2D = means2D[valid]
            valid_opacity = opacity[valid]
            valid_shs = shs[valid]
            valid_scales = scales[valid]
            valid_rots = rots[valid]
            for r_idx, rotation in enumerate(rotations):
                c2w = rotation
                c2w[:3, 3] = position
                w2c = torch.inverse(c2w)
                T = w2c[:3, 3]
                R = w2c[:3, :3].T
                world_view_transform = getWorld2ViewTorch(R, T).transpose(0, 1)
                full_proj_transform = (
                    world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
                ).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]

                input_args = (
                    bg_color,
                    # bg_colors[r_idx],
                    valid_means3D,
                    torch.Tensor([]),
                    valid_opacity,
                    valid_scales,
                    valid_rots,
                    torch.Tensor([]),
                    shs,
                    camera_center,  # campos,
                    world_view_transform,  # viewmatrix,
                    full_proj_transform,  # projmatrix,
                    1.0,  # scale_modifier
                    1.0,  # tanfovx,
                    1.0,  # tanfovy,
                    res,  # image_height,
                    res,  # image_width,
                    gaussians.active_sh_degree,
                    False,  # prefiltered,
                    False,  # argmax_depth,
                )
                (num_rendered, rendered_image, opacity_map, radii, depth_map) = _C.lite_rasterize_gaussians(
                    *input_args
                )
                rgb_cubemap.append(rendered_image.permute(1, 2, 0))
                opacity_cubemap.append(opacity_map.permute(1, 2, 0))
                depth_map = depth_map * (opacity_map > 0.5).float()  # NOTE: import to filter out the floater
                depth_cubemap.append(depth_map.permute(1, 2, 0) * norm)

            # convert cubemap to HDRI
            depth_envmap = dr.texture(
                torch.stack(depth_cubemap)[None, ...],
                envmap_dirs[None, ...].contiguous(),
                # filter_mode="linear",
                filter_mode="nearest",
                boundary_mode="cube",
            )[
                0
            ]  # [H, W, 1]

            # use SH to store the HDRI
            occlu_mask = (1 - (depth_envmap < occlusion_threshold).float()) + (depth_envmap == 0).float()  # [H, W, 1]

            weighted_color = occlu_mask * solid_angles  # [H, W, 1]
            temp_coefficients = (weighted_color * components).sum(0).sum(0)  # [d2]
            occlusion_coefficients[grid_id] = temp_coefficients[:, None]

        # dialate coefficient ids
        while (occlusion_ids == -1).sum() > 0:
            gs_ir_ext.dialate_occlusion_ids(occlusion_ids)

    save_file = os.path.join(os.path.dirname(args.checkpoint), "occlusion_volumes.pth")
    torch.save(
        {
            "occlusion_ids": occlusion_ids,
            "occlusion_coefficients": occlusion_coefficients,
            "bound": args.bound,
            "degree": occlu_sh_degree,
            "occlusion_threshold": occlusion_threshold,
        },
        save_file,
    )
    print(f"save occlusion volumes as {save_file}")

