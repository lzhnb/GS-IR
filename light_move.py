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

import os
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple, Union

import cv2
import imageio.v2 as imageio
import numpy as np
import nvdiffrast.torch as dr
import open3d as o3d
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from diff_gaussian_rasterization import _C
from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene, Camera
from utils.graphics_utils import getProjectionMatrix
from utils.camera_utils import trajectory_from_c2ws
from utils.image_utils import turbo_cmap


def saturate_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)


def DistributionGGX(
    normals: torch.Tensor,  # [H, W, 3]
    half_dirs: torch.Tensor,  # [H, W, 3]
    roughness: torch.Tensor,  # [H, W, 1]
) -> torch.Tensor:
    a = roughness * roughness
    a2 = a * a
    NoH = saturate_dot(normals, half_dirs)
    NoH2 = NoH * NoH

    nom = a2
    denom = (NoH2 * (a2 - 1.0) + 1.0)
    denom = np.pi * denom * denom

    return nom / denom


def GeometrySchlickGGX(
    NoV: torch.Tensor, # [H, W, 1]
    roughness: torch.Tensor,  # [H, W, 1]
) -> torch.Tensor:
    r = roughness + 1.0
    k = (r * r) / 8.0
    nom = NoV
    denom = NoV * (1.0 - k) + k

    return nom / denom

def GeometrySmith(
    normals: torch.Tensor,  # [H, W, 3]
    view_dirs: torch.Tensor,  # [H, W, 3]
    light_dirs: torch.Tensor,  # [H, W, 3]
    roughness: torch.Tensor,  # [H, W, 1]
) -> torch.Tensor:
    NoV = saturate_dot(normals, view_dirs)
    NoL = saturate_dot(normals, light_dirs)
    ggx2 = GeometrySchlickGGX(NoV, roughness)
    ggx1 = GeometrySchlickGGX(NoL, roughness)

    return ggx1 * ggx2


def fresnelSchlick(
    HoV: torch.Tensor,  # [H, W, 1]
    F0: torch.Tensor,  # [H, W, 3]
) -> torch.Tensor:
    return F0 + (1.0 - F0) * torch.pow((1.0 - HoV).clamp(0.0, 1.0), 5)


def linear_to_srgb(linear: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError


# https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/6.pbr/2.2.1.ibl_specular/2.2.1.pbr.fs
def light_pbr_shading(
    light_position: torch.Tensor,  # [3]
    light_intensity: torch.Tensor,  # [3]
    points: torch.Tensor,  # [H, W, 3]
    normals: torch.Tensor,  # [H, W, 3]
    view_dirs: torch.Tensor,  # [H, W, 3]
    albedo: torch.Tensor,  # [H, W, 3]
    roughness: torch.Tensor,  # [H, W, 1]
    mask: torch.Tensor,  # [H, W, 1]
    linear: bool = False,
    metallic: Optional[torch.Tensor] = None,
    shadow: Optional[torch.Tensor] = None,
    background: Optional[torch.Tensor] = None,
) -> Dict:
    if background is None:
        background = torch.zeros_like(normals)  # [H, W, 3]

    # preapre
    light_dirs = F.normalize(light_position - points, p=2, dim=-1)  # [H, W, 3]
    half_dirs = (light_dirs + view_dirs) / 2.0  # [H, W, 3]
    distance = torch.norm(light_position - points, p=2, dim=-1, keepdim=True)  # [H, W, 1]
    attenuation = 1.0 / torch.pow(distance, 2)  # [H, W, 1]
    radiance = light_intensity * attenuation  # [H, W, 3]

    if metallic is None:
        F0 = torch.ones_like(albedo) * 0.04  # [H, W, 3]
    else:
        F0 = (1.0 - metallic) * 0.04 + albedo * metallic  # [H, W, 3]

    # Cook-Torrance BRDF
    NoV = saturate_dot(normals, view_dirs)  # [H, W, 1]
    NoL = saturate_dot(normals, light_dirs)  # [H, W, 1]
    HoV = saturate_dot(half_dirs, view_dirs)  # [H, W, 1]
    NDF = DistributionGGX(normals=normals, half_dirs=half_dirs, roughness=roughness)  # [H, W, 1]
    G = GeometrySmith(normals=normals, view_dirs=view_dirs, light_dirs=light_dirs, roughness=roughness)  # [H, W, 1]
    fresnel = fresnelSchlick(HoV=HoV, F0=F0)  # [H, W, 3]

    numerator = NDF * G * fresnel  # [H, W, 3]
    denominator = 4.0 * NoV * NoL + 1e-4  # [H, W, 1]
    specular = numerator / denominator  # [H, W, 3]

    kd = 1.0 - fresnel  # [H, W, 3]
    if metallic is not None:
        kd *= (1.0 - metallic)
    
    render_rgb = (kd * albedo / np.pi + specular) * radiance * NoL

    render_rgb = torch.where(mask, render_rgb, background)

    if shadow is not None:
        render_rgb = torch.where(shadow == 0.0, render_rgb, render_rgb * 0.2)

    if linear:
        render_rgb = linear_to_srgb(render_rgb.squeeze())

    results = {}
    results.update(
        {
            "render_rgb": render_rgb,
        }
    )

    return results


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


def getWorld2ViewTorch(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    Rt = torch.zeros((4, 4), device=R.device)
    Rt[:3, :3] = R[:3, :3].T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


# inverse the mapping from https://github.com/NVlabs/nvdiffrec/blob/dad3249af8ede96c7dd72c30328272117fabb710/render/light.py#L22
def get_envmap_dirs(res: List[int] = [256, 512]) -> torch.Tensor:
    gy, gx = torch.meshgrid(
        torch.linspace(0.0, 1.0 - 1.0 / res[0], res[0], device="cuda"),
        torch.linspace(-1.0, 1.0 - 1.0 / res[1], res[1], device="cuda"),
        indexing="ij",
    )

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]

    return reflvec


def get_depth_cubemap(
    gaussians: GaussianModel, position: torch.Tensor, res: int = 2048
) -> torch.Tensor:
    # get canonical ray and its norm to normalize depth
    canonical_rays = get_canonical_rays(H=res, W=res, tan_fovx=1.0, tan_fovy=1.0)  # [HW, 3]
    norm = torch.norm(canonical_rays, p=2, dim=-1).reshape(res, res, 1)  # [H, W]

    bg_color = torch.zeros([3, res, res], device="cuda")
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
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=np.pi * 0.5, fovY=np.pi * 0.5)
        .transpose(0, 1)
        .cuda()
    )

    depth_cubemap = []
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
            gaussians.get_xyz,
            torch.Tensor([]),
            gaussians.get_opacity,
            gaussians.get_scaling,
            gaussians.get_rotation,
            torch.Tensor([]),
            gaussians.get_features,
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
        (num_rendered, rendered_image, opacity_map, radii, depth_map) = _C.lite_rasterize_gaussians(*input_args)
        # depth_cubemap.append(depth_map.permute(1, 2, 0) * norm)
        depth_cubemap.append(depth_map.permute(1, 2, 0))

    return torch.stack(depth_cubemap)


@torch.no_grad()
def launch(
    model_path: str,
    checkpoint: str,
    dataset: GroupParams,
    pipeline: GroupParams,
    frames: int,
    fps: int,
    metallic: bool = False,
    linear: bool = False,
    argmax_depth: bool = False,
    start: int = -1,
    end: int = -1,
    loop: bool = False,
) -> None:
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    checkpoint = torch.load(checkpoint)
    if isinstance(checkpoint, Tuple):
        model_params = checkpoint[0]
    elif isinstance(checkpoint, Dict):
        model_params = checkpoint["gaussians"]
    else:
        raise TypeError

    gaussians.restore(model_params)

    views = scene.getTrainCameras()

    name = "light_move"
    pbr_path = os.path.join(model_path, name, "pbr")
    depth_path = os.path.join(model_path, name, "pbr")
    os.makedirs(pbr_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)

    canonical_rays = scene.get_canonical_rays()

    # generate trajectory
    c2ws = []
    if start == -1:
        start = 0
    if end == -1:
        end = len(views) - 1
    assert end > start
    for i in range(start, end):
        c2w = torch.inverse(views[i].world_view_transform.T).cpu().numpy()  # [4, 4]
        c2ws.append(c2w)
    if loop:
        c2ws.append(c2ws[0])

    # interpolate c2w according to frames
    c2ws_inter = trajectory_from_c2ws(c2ws=c2ws, frames=frames)

    # NOTE: get the reference view and only change its `world_view_transform` and `camera_center` according to c2w_inter
    ref_view = views[start]
    H, W = ref_view.image_height, ref_view.image_width
    norm = torch.norm(canonical_rays, p=2, dim=-1).reshape(H, W, 1)  # [H, W]
    # write video
    shadow_video_writer = cv2.VideoWriter(
        filename=os.path.join(model_path, name, "light_move.mp4"),
        fourcc=cv2.VideoWriter_fourcc(*"MJPG"),
        fps=fps,
        frameSize=(ref_view.image_width, ref_view.image_height),
    )
    
    background = ref_view.bg_color.cuda()
    rendering_result = render(
        viewpoint_camera=ref_view,
        pc=scene.gaussians,
        pipe=pipeline,
        bg_color=background,
        inference=True,
        pad_normal=True,
        derive_normal=True,
        argmax_depth=argmax_depth,
    )

    render_img = rendering_result["render"]
    depth_map = rendering_result["depth_map"]
    normal_map = rendering_result["normal_map"]
    normal_mask = rendering_result["normal_mask"]
    albedo_map = rendering_result["albedo_map"]  # [3, H, W]
    roughness_map = rendering_result["roughness_map"]  # [1, H, W]
    metallic_map = rendering_result["metallic_map"]  # [1, H, W]

    c2w = torch.inverse(ref_view.world_view_transform.T)  # [4, 4]
    view_dirs = -(
        (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
        .sum(dim=-1)
        .reshape(H, W, 3)
    )  # [H, W, 3]
    points = (
        -view_dirs.reshape(-1, 3) * norm.reshape(-1, 1) * depth_map.reshape(-1, 1) + c2w[:3, 3]
    ).contiguous()  # [HW, 3]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pc.colors = o3d.utility.Vector3dVector(albedo_map.permute(1, 2, 0).reshape(-1, 3).cpu().numpy())
    o3d.io.write_point_cloud(os.path.join(model_path, name, "ref_pc.ply"), pc)

    # set point light source
    light_intensity = torch.ones([3]).cuda() * 100.0
    envmap_dirs = get_envmap_dirs()  # [H, W, 3]
    for idx, c2w_inter in enumerate(tqdm(c2ws_inter, desc="Rendering progress")):
        idx = 120
        c2w_inter = c2ws_inter[idx]
        c2w_inter = torch.from_numpy(c2w_inter).cuda().float()
        light_position = c2w_inter[:3, 3]

        # get depth cubemap at light source
        depth_cubemap = get_depth_cubemap(
            gaussians=gaussians, position=light_position
        )  # [6, res, res, 1]
        depth_envmap = dr.texture(
            depth_cubemap[None, ...],
            envmap_dirs[None, ...].contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )[
            0
        ]  # [H, W, 1]
        depth_envmap_img = (turbo_cmap(depth_envmap.squeeze().cpu().numpy()) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(depth_path, f"{idx:05d}_depth_cubemap.png"), depth_envmap_img)

        # for debug
        depth_points = (envmap_dirs * depth_envmap).reshape(
            -1, 3
        ).cpu().numpy() + light_position.cpu().numpy()
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(depth_points)
        o3d.io.write_point_cloud(os.path.join(model_path, name, "depth_cubemap.ply"), pc)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(light_position.reshape(1, 3).cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(model_path, name, "light.ply"), pc)

        to_light = (light_position[None, ...] - points).reshape(H, W, 3)  # [H, W, 3]
        distance_to_light = torch.norm(to_light, p=2, dim=-1).reshape(H, W, 1)
        query_dirs = F.normalize(-to_light, p=2, dim=-1)  # [H, W, 3]
        closest_depth = dr.texture(
            depth_cubemap[None, ...],
            query_dirs[None, ...].contiguous(),
            filter_mode="linear",
            # filter_mode="nearest",
            boundary_mode="cube",
        )[
            0
        ]  # [H, W, 1]
        threshold = 2e-1
        shadow = (distance_to_light - threshold > closest_depth).float().permute(2, 0, 1)
        img = torch.cat([render_img, torch.tile(shadow, (3, 1, 1))], dim=2)
        torchvision.utils.save_image(img, os.path.join(pbr_path, f"{idx:05d}_shadow.png"))

        depth = torch.cat([distance_to_light, closest_depth], dim=1).squeeze().cpu().numpy()
        depth_img = (turbo_cmap(depth) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(pbr_path, f"{idx:05d}_depth.png"), depth_img)

        pbr_result = light_pbr_shading(
            light_position=light_position,
            light_intensity=light_intensity,
            points=points.reshape(H, W, 3),
            normals=normal_map.permute(1, 2, 0),  # [H, W, 3]
            view_dirs=view_dirs,
            mask=normal_mask.permute(1, 2, 0),  # [H, W, 1]
            albedo=albedo_map.permute(1, 2, 0),  # [H, W, 3]
            roughness=roughness_map.permute(1, 2, 0),  # [H, W, 1]
            metallic=metallic_map.permute(1, 2, 0) if metallic else None,  # [H, W, 1]
            shadow = shadow.permute(1, 2, 0),  # [H, W, 1]
            linear=linear,
        )
        render_rgb = (
            pbr_result["render_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
        )  # [3, H, W]
        torchvision.utils.save_image(render_rgb, os.path.join(pbr_path, f"{idx:05d}.png"))
        img = cv2.imread(os.path.join(pbr_path, f"{idx:05d}.png"))
        shadow_video_writer.write(img)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--frames", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--metallic", action="store_true")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--argmax_depth", action="store_true")
    args = get_combined_args(parser)
    args.eval = False

    model_path = os.path.dirname(args.checkpoint)
    print("Rendering " + model_path)

    launch(
        model_path=model_path,
        checkpoint=args.checkpoint,
        dataset=model.extract(args),
        pipeline=pipeline.extract(args),
        frames=args.frames,
        fps=args.fps,
        argmax_depth=args.argmax_depth,
        metallic=args.metallic,
        linear=args.linear,
        start=args.start,
        end=args.end,
        loop=args.loop,
    )

# python light_move.py -m output/garden-linear/ -s dataset/nerf_data/nerf_real_360/garden/ --checkpoint output/garden-linear/chkpnt35000.pth --frames 240 --fps 30 --start 158 --end 184 --loop --linear
