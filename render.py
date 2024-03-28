import os
import json
from argparse import ArgumentParser
from typing import Dict, Optional

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from PIL import Image
from lpips import LPIPS

from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from gs_ir import recon_occlusion, IrradianceVolumes
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import viridis_cmap, psnr as get_psnr
from utils.loss_utils import ssim as get_ssim


def render_set(
    model_path: str,
    name: str,
    scene: Scene,
    light: CubemapLight,
    irradiance_volumes: IrradianceVolumes,
    pipeline: GroupParams,
    occlusion_volumes: Optional[Dict] = None,
    pbr: bool = False,
    metallic: bool = False,
    tone: bool = False,
    gamma: bool = False,
    indirect: bool = False,
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
    envmap_path = os.path.join(model_path, name, "envmap.png")
    torchvision.utils.save_image(envmap, envmap_path)

    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    depths_path = os.path.join(model_path, name, f"ours_{iteration}", "depth")
    normals_path = os.path.join(model_path, name, f"ours_{iteration}", "normal")
    pbr_path = os.path.join(model_path, name, f"ours_{iteration}", "pbr")
    pc_path = os.path.join(model_path, name, f"ours_{iteration}", "pc")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(depths_path, exist_ok=True)
    os.makedirs(normals_path, exist_ok=True)
    os.makedirs(pbr_path, exist_ok=True)
    os.makedirs(pc_path, exist_ok=True)

    brdf_lut = get_brdf_lut().cuda()
    canonical_rays = scene.get_canonical_rays()

    ref_view = views[0]
    H, W = ref_view.image_height, ref_view.image_width
    c2w = torch.inverse(ref_view.world_view_transform.T)  # [4, 4]
    view_dirs_ = (  # NOTE: no negative here
        (canonical_rays[:, None, :] * c2w[None, :3, :3]).sum(dim=-1).reshape(H, W, 3)  # [HW, 3, 3]
    )  # [H, W, 3]
    norm = torch.norm(canonical_rays, p=2, dim=-1).reshape(H, W, 1)


    psnr_avg = 0.0
    ssim_avg = 0.0
    lpips_avg = 0.0
    lpips_fn = LPIPS(net="vgg").cuda()

    if occlusion_volumes is not None:
        occlusion_ids = occlusion_volumes["occlusion_ids"]
        occlusion_coefficients = occlusion_volumes["occlusion_coefficients"]
        occlusion_degree = occlusion_volumes["degree"]
        bound = occlusion_volumes["bound"]
        aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        rendering_result = render(
            viewpoint_camera=view,
            pc=scene.gaussians,
            pipe=pipeline,
            bg_color=background,
            inference=True,
            pad_normal=True,
            derive_normal=True,
        )

        gt_image = view.original_image.cuda()
        alpha_mask = view.gt_alpha_mask.cuda()
        gt_image = (gt_image * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)
        depth_map = rendering_result["depth_map"]

        depth_img = viridis_cmap(depth_map.squeeze().cpu().numpy())
        depth_img = (depth_img * 255).astype(np.uint8)
        normal_map_from_depth = rendering_result["normal_map_from_depth"]
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

        if indirect:
            points = (
                (-view_dirs.reshape(-1, 3) * depth_map.reshape(-1, 1) + c2w[:3, 3])
                .clamp(min=-bound, max=bound)
                .contiguous()
            )  # [HW, 3]
            occlusion = recon_occlusion(
                H=H,
                W=W,
                points=points,
                normals=normal_map.permute(1, 2, 0).reshape(-1, 3).contiguous(),
                bound=bound,
                occlusion_coefficients=occlusion_coefficients,
                occlusion_ids=occlusion_ids,
                aabb=aabb,
                degree=occlusion_degree,
            ).reshape(H, W, 1)
            irradiance = irradiance_volumes.query_irradiance(
                points=points.reshape(-1, 3).contiguous(),
                normals=normal_map.permute(1, 2, 0).reshape(-1, 3).contiguous(),
            ).reshape(H, W, -1)
        else:
            occlusion = torch.ones_like(depth_map).permute(1, 2, 0)  # [H, W, 1]
            irradiance = torch.zeros_like(depth_map).permute(1, 2, 0)  # [H, W, 1]

        torchvision.utils.save_image(
            (normal_map + 1) / 2, os.path.join(normals_path, f"{idx:05d}_normal.png")
        )
        torchvision.utils.save_image(
            (normal_map_from_depth + 1) / 2,
            os.path.join(normals_path, f"{idx:05d}_from_depth.png"),
        )

        if pbr:
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
                occlusion=occlusion,
                irradiance=irradiance,
                brdf_lut=brdf_lut,
            )
            render_rgb = (
                pbr_result["render_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
            )  # [3, H, W]
            background = torch.zeros_like(render_rgb)
            render_rgb = torch.where(
                normal_mask,
                render_rgb,
                background,
            )
            brdf_map = torch.cat(
                [
                    albedo_map,
                    torch.tile(roughness_map, (3, 1, 1)),
                    torch.tile(metallic_map, (3, 1, 1)),
                ],
                dim=2,
            )  # [3, H, 3W]
            torchvision.utils.save_image(brdf_map, os.path.join(pbr_path, f"{idx:05d}_brdf.png"))
            torchvision.utils.save_image(render_rgb, os.path.join(pbr_path, f"{idx:05d}.png"))

            psnr_avg += get_psnr(gt_image, render_rgb).mean().double()
            ssim_avg += get_ssim(gt_image, render_rgb).mean().double()
            lpips_avg += lpips_fn(gt_image, render_rgb).mean().double()

    if pbr:
        psnr = psnr_avg / len(views)
        ssim = ssim_avg / len(views)
        lpips = lpips_avg / len(views)
        print(f"psnr_avg: {psnr}; ssim_avg: {ssim}; lpips_avg: {lpips}")


@torch.no_grad()
def launch(
    model_path: str,
    checkpoint_path: str,
    dataset: GroupParams,
    pipeline: GroupParams,
    skip_train: bool,
    skip_test: bool,
    pbr: bool = False,
    metallic: bool = False,
    tone: bool = False,
    gamma: bool = False,
    indirect: bool = False,
    brdf_eval: bool = False,
) -> None:
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    cubemap = CubemapLight(base_res=256).cuda()

    # occlusion volumes
    filepath = os.path.join(os.path.dirname(checkpoint_path), "occlusion_volumes.pth")
    print(f"begin to load occlusion volumes from {filepath}")
    if os.path.exists(filepath):
        occlusion_volumes = torch.load(filepath)
        bound = occlusion_volumes["bound"]
    else:
        occlusion_volumes = None
        bound = 0.5
    aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
    irradiance_volumes = IrradianceVolumes(aabb=aabb).cuda()

    checkpoint = torch.load(checkpoint_path)
    model_params = checkpoint["gaussians"]
    cubemap_params = checkpoint["cubemap"]
    irradiance_volumes_params = checkpoint["irradiance_volumes"]

    gaussians.restore(model_params)
    cubemap.load_state_dict(cubemap_params)
    cubemap.eval()
    irradiance_volumes.load_state_dict(irradiance_volumes_params)
    irradiance_volumes.eval()

    if brdf_eval:
        if not skip_train:
            eval_brdf(
                data_root=dataset.source_path,
                scene=scene,
                model_path=model_path,
                name="train",
            )
        if not skip_test:
            eval_brdf(
                data_root=dataset.source_path,
                scene=scene,
                model_path=model_path,
                name="test",
            )
    else:
        if not skip_train:
            render_set(
                model_path=model_path,
                name="train",
                scene=scene,
                light=cubemap,
                irradiance_volumes=irradiance_volumes,
                occlusion_volumes=occlusion_volumes,
                pipeline=pipeline,
                pbr=pbr,
                metallic=metallic,
                tone=tone,
                gamma=gamma,
                indirect=indirect,
            )
        if not skip_test:
            render_set(
                model_path=model_path,
                name="test",
                scene=scene,
                light=cubemap,
                irradiance_volumes=irradiance_volumes,
                occlusion_volumes=occlusion_volumes,
                pipeline=pipeline,
                pbr=pbr,
                metallic=metallic,
                tone=tone,
                gamma=gamma,
                indirect=indirect,
            )


def eval_brdf(data_root: str, scene: Scene, model_path: str, name: str) -> None:
    # only for TensoIR synthetic
    if name == "train":
        transform_file = os.path.join(data_root, "transforms_train.json")
    elif name == "test":
        transform_file = os.path.join(data_root, "transforms_test.json")

    with open(transform_file, "r") as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]

    iteration = scene.loaded_iter
    pbr_dir = os.path.join(model_path, name, f"ours_{iteration}", "pbr")

    albedo_psnr_avg = 0.0
    albedo_ssim_avg = 0.0
    albedo_lpips_avg = 0.0

    pbr_path = os.path.join(model_path, name, f"ours_{iteration}", "pbr")
    albedo_gts = []
    albedo_maps = []
    masks = []
    gt_albedo_list = []
    reconstructed_albedo_list = []
    lpips_fn = LPIPS(net="vgg").cuda()
    for idx, frame in enumerate(tqdm(frames)):
        # read gt
        albedo_path = frame["file_path"].replace("rgba", "albedo") + ".png"
        albedo_gt = np.array(Image.open(os.path.join(data_root, albedo_path)))[..., :3]
        mask = np.array(Image.open(os.path.join(data_root, albedo_path)))[..., 3] > 0
        albedo_gt = torch.from_numpy(albedo_gt).cuda() / 255.0  # [H, W, 3]
        albedo_gts.append(albedo_gt)
        mask = torch.from_numpy(mask).cuda()  # [H, W]
        masks.append(mask)
        gt_albedo_list.append(albedo_gt[mask])
        # read prediction
        brdf_map = np.array(Image.open(os.path.join(pbr_dir, f"{idx:05}_brdf.png")))
        H, W3, _ = brdf_map.shape
        albedo_map = brdf_map[:, : (W3 // 3), :]  # [H, W, 3]
        albedo_map = torch.from_numpy(albedo_map).cuda() / 255.0  # [H, W, 3]
        albedo_maps.append(albedo_map)
        reconstructed_albedo_list.append(albedo_map[mask])
    gt_albedo_all = torch.cat(gt_albedo_list, dim=0)
    albedo_map_all = torch.cat(reconstructed_albedo_list, dim=0)
    # single_channel_ratio = (gt_albedo_all / albedo_map_all.clamp(min=1e-6))[..., 0].median()  # [1]
    three_channel_ratio, _ = (gt_albedo_all / albedo_map_all.clamp(min=1e-6)).median(dim=0)  # [3]

    for idx, (mask, albedo_map, albedo_gt) in enumerate(tqdm(zip(masks, albedo_maps, albedo_gts))):
        albedo_map[mask] *= three_channel_ratio
        albedo_map = albedo_map.permute(2, 0, 1)  # [3, H, W]
        albedo_gt = albedo_gt.permute(2, 0, 1)  # [3, H, W]
        torchvision.utils.save_image(albedo_map, os.path.join(pbr_path, f"{idx:05d}_albedo.png"))
        albedo_psnr_avg += get_psnr(albedo_gt, albedo_map).mean().double()
        albedo_ssim_avg += get_ssim(albedo_gt, albedo_map).mean().double()
        albedo_lpips_avg += lpips_fn(albedo_gt, albedo_map).mean().double()

    albedo_psnr = albedo_psnr_avg / len(frames)
    albedo_ssim = albedo_ssim_avg / len(frames)
    albedo_lpips = albedo_lpips_avg / len(frames)
    print(f"albedo psnr_avg: {albedo_psnr}; ssim_avg: {albedo_ssim}; lpips_avg: {albedo_lpips}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None, help="The path to the checkpoint to load.")
    parser.add_argument("--pbr", action="store_true", help="Enable pbr rendering for NVS evaluation and export BRDF map.")
    parser.add_argument("--tone", action="store_true", help="Enable aces film tone mapping.")
    parser.add_argument("--gamma", action="store_true", help="Enable linear_to_sRGB for gamma correction.")
    parser.add_argument("--metallic", action="store_true", help="Enable metallic material reconstruction.")
    parser.add_argument("--indirect", action="store_true", help="Enable indirect diffuse modeling.")
    parser.add_argument("--brdf_eval", action="store_true", help="Enable to evaluate reconstructed BRDF.")
    args = get_combined_args(parser)

    model_path = os.path.dirname(args.checkpoint)
    print("Rendering " + model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    launch(
        model_path=model_path,
        checkpoint_path=args.checkpoint,
        dataset=model.extract(args),
        pipeline=pipeline.extract(args),
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        pbr=args.pbr,
        metallic=args.metallic,
        tone=args.tone,
        gamma=args.gamma,
        indirect=args.indirect,
        brdf_eval=args.brdf_eval,
    )
