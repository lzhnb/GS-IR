import os
from argparse import ArgumentParser
from os import makedirs
from typing import Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import viridis_cmap
import nvdiffrast.torch as dr


def read_hdr(path: str) -> np.ndarray:
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    with open(path, "rb") as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap(latlong_map: torch.Tensor, res: List[int]) -> torch.Tensor:
    cubemap = torch.zeros(
        6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device="cuda"
    )
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
            indexing="ij",
        )
        v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(
            latlong_map[None, ...], texcoord[None, ...], filter_mode="linear"
        )[0]
    return cubemap


def render_set(
    model_path: str,
    name: str,
    light_name: str,
    scene: Scene,
    hdri: torch.Tensor,
    light: CubemapLight,
    pipeline: GroupParams,
    metallic: bool = False,
    tone: bool = False,
    gamma: bool = False,
) -> None:
    iteration = scene.loaded_iter
    if name == "train":
        views = scene.getTrainCameras()
    elif name == "test":
        views = scene.getTestCameras()
    else:
        raise ValueError

    # build mip for environment light
    light.build_mips()
    envmap = light.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
    os.makedirs(os.path.join(model_path, name), exist_ok=True)
    envmap_path = os.path.join(model_path, name, "envmap_relight.png")
    torchvision.utils.save_image(envmap, envmap_path)

    relight_path = os.path.join(model_path, name, f"ours_{iteration}", "relight")
    makedirs(relight_path, exist_ok=True)

    brdf_lut = get_brdf_lut().cuda()
    canonical_rays = scene.get_canonical_rays()

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        background[...] = 0.0  # NOTE: set zero
        rendering_result = render(
            viewpoint_camera=view,
            pc=scene.gaussians,
            pipe=pipeline,
            bg_color=background,
            inference=True,
            pad_normal=True,
            derive_normal=True,
        )

        depth_map = rendering_result["depth_map"]

        depth_img = viridis_cmap(depth_map.squeeze().cpu().numpy())
        depth_img = (depth_img * 255).astype(np.uint8)
        normal_map = rendering_result["normal_map"]
        normal_mask = rendering_result["normal_mask"]

        # normal from point cloud
        H, W = view.image_height, view.image_width
        c2w = torch.inverse(view.world_view_transform.T)  # [4, 4]
        view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
            .sum(dim=-1)
            .reshape(H, W, 3)
        )  # [H, W, 3]
        alpha_mask = view.gt_alpha_mask.cuda()

        albedo_map = rendering_result["albedo_map"]  # [3, H, W]
        roughness_map = rendering_result["roughness_map"]  # [1, H, W]
        metallic_map = rendering_result["metallic_map"]  # [1, H, W]
        pbr_result = pbr_shading(
            light=light,
            normals=normal_map.permute(1, 2, 0),  # [H, W, 3]
            view_dirs=view_dirs,
            mask=normal_mask.permute(1, 2, 0),  # [H, W, 1]
            albedo=albedo_map.permute(1, 2, 0),  # [H, W, 3]
            roughness=roughness_map.permute(1, 2, 0),  # [H, W, 1]
            metallic=metallic_map.permute(1, 2, 0) if metallic else None,  # [H, W, 1]
            tone=tone,
            gamma=gamma,
            brdf_lut=brdf_lut,
        )
        render_rgb = pbr_result["render_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)  # [3, H, W]

        render_rgb = render_rgb * alpha_mask

        torchvision.utils.save_image(
            render_rgb, os.path.join(relight_path, f"{idx:05d}_{light_name}.png")
        )


@torch.no_grad()
def launch(
    model_path: str,
    checkpoint: str,
    hdri_path: str,
    dataset: GroupParams,
    pipeline: GroupParams,
    skip_train: bool,
    skip_test: bool,
    metallic: bool = False,
    tone: bool = False,
    gamma: bool = False,
) -> None:
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    # load hdri
    print(f"read hdri from {hdri_path}")
    hdri = read_hdr(hdri_path)
    hdri = torch.from_numpy(hdri).cuda()
    res = 256
    cubemap = CubemapLight(base_res=res).cuda()
    cubemap.base.data = latlong_to_cubemap(hdri, [res, res])
    cubemap.eval()

    light_name = os.path.basename(hdri_path).split(".")[0]

    checkpoint = torch.load(checkpoint)
    if isinstance(checkpoint, Tuple):
        model_params = checkpoint[0]
    elif isinstance(checkpoint, Dict):
        model_params = checkpoint["gaussians"]
    else:
        raise TypeError
    gaussians.restore(model_params)

    if not skip_train:
        render_set(
            model_path=model_path,
            name="train",
            light_name=light_name,
            scene=scene,
            hdri=hdri,
            light=cubemap,
            metallic=metallic,
            tone=tone,
            gamma=gamma,
            pipeline=pipeline,
        )
    if not skip_test:
        render_set(
            model_path=model_path,
            name="test",
            light_name=light_name,
            scene=scene,
            hdri=hdri,
            light=cubemap,
            metallic=metallic,
            tone=tone,
            gamma=gamma,
            pipeline=pipeline,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--hdri", type=str, default=None, help="The path to the hdri for relighting.")
    parser.add_argument("--checkpoint", type=str, default=None, help="The path to the checkpoint to load.")
    parser.add_argument("--tone", action="store_true", help="Enable aces film tone mapping.")
    parser.add_argument("--gamma", action="store_true", help="Enable linear_to_sRGB for gamma correction.")
    parser.add_argument("--metallic", action="store_true", help="Enable metallic material reconstruction.")
    args = get_combined_args(parser)

    model_path = os.path.dirname(args.checkpoint)
    print("Rendering " + model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    launch(
        model_path=model_path,
        checkpoint=args.checkpoint,
        hdri_path=args.hdri,
        dataset=model.extract(args),
        pipeline=pipeline.extract(args),
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        metallic=args.metallic,
        tone=args.tone,
        gamma=args.gamma,
    )


