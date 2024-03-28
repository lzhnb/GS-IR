import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Dict, List, Optional, Tuple, Union

import kornia
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm, trange

from arguments import GroupParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from gs_ir import recon_occlusion, IrradianceVolumes
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from scene import GaussianModel, Scene, Camera
from utils.general_utils import safe_state
from utils.image_utils import psnr, turbo_cmap
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def get_tv_loss(
    gt_image: torch.Tensor,  # [3, H, W]
    prediction: torch.Tensor,  # [C, H, W]
    pad: int = 1,
    step: int = 1,
) -> torch.Tensor:
    if pad > 1:
        gt_image = F.avg_pool2d(gt_image, pad, pad)
        prediction = F.avg_pool2d(prediction, pad, pad)
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]
    tv_loss = (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    if step > 1:
        for s in range(2, step + 1):
            rgb_grad_h = torch.exp(
                -(gt_image[:, s:, :] - gt_image[:, :-s, :]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            rgb_grad_w = torch.exp(
                -(gt_image[:, :, s:] - gt_image[:, :, :-s]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            tv_h = torch.pow(prediction[:, s:, :] - prediction[:, :-s, :], 2)  # [C, H-1, W]
            tv_w = torch.pow(prediction[:, :, s:] - prediction[:, :, :-s], 2)  # [C, H, W-1]
            tv_loss += (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    return tv_loss


def get_masked_tv_loss(
    mask: torch.Tensor,  # [1, H, W]
    gt_image: torch.Tensor,  # [3, H, W]
    prediction: torch.Tensor,  # [C, H, W]
    erosion: bool = False,
) -> torch.Tensor:
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]

    # erode mask
    mask = mask.float()
    if erosion:
        kernel = mask.new_ones([7, 7])
        mask = kornia.morphology.erosion(mask[None, ...], kernel)[0]
    mask_h = mask[:, 1:, :] * mask[:, :-1, :]  # [1, H-1, W]
    mask_w = mask[:, :, 1:] * mask[:, :, :-1]  # [1, H, W-1]

    tv_loss = (tv_h * rgb_grad_h * mask_h).mean() + (tv_w * rgb_grad_w * mask_w).mean()

    return tv_loss


def get_envmap_dirs(res: List[int] = [512, 1024]) -> torch.Tensor:
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
        torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
        indexing="ij",
    )

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]
    return reflvec


def resize_tensorboard_img(
    img: torch.Tensor,  # [C, H, W]
    max_res: int = 800,
) -> torch.Tensor:
    _, H, W = img.shape
    ratio = min(max_res / H, max_res / W)
    target_size = (int(H * ratio), int(W * ratio))
    transform = T.Resize(size=target_size)
    img = transform(img)  # [C, H', W']
    return img


def training(
    dataset: GroupParams,
    opt: GroupParams,
    pipe: GroupParams,
    testing_iterations: List[int],
    saving_iterations: List[int],
    checkpoint_iterations: int,
    checkpoint_path: Optional[str] = None,
    pbr_iteration: int = 30_000,
    debug_from: int = -1,
    metallic: bool = False,
    tone: bool = False,
    gamma: bool = False,
    normal_tv_weight: float = 1.0,
    brdf_tv_weight: float = 1.0,
    env_tv_weight: float = 0.01,
    bound: float = 1.5,
    indirect: bool = False,
) -> None:
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    tb_writer = prepare_output_and_logger(dataset)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # NOTE: prepare for PBR
    brdf_lut = get_brdf_lut().cuda()
    envmap_dirs = get_envmap_dirs()
    cubemap = CubemapLight(base_res=256).cuda()
    cubemap.train()
    aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
    irradiance_volumes = IrradianceVolumes(aabb=aabb).cuda()
    irradiance_volumes.train()
    param_groups = [
        {
            "name": "irradiance_volumes",
            "params": irradiance_volumes.parameters(),
            "lr": opt.opacity_lr,
        },
        {"name": "cubemap", "params": cubemap.parameters(), "lr": opt.opacity_lr},
    ]
    light_optimizer = torch.optim.Adam(param_groups, lr=opt.opacity_lr)

    canonical_rays = scene.get_canonical_rays()

    # load checkpoint
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model_params = checkpoint["gaussians"]
        first_iter = checkpoint["iteration"]
        # cubemap_params = checkpoint["cubemap"]
        # light_optimizer_params = checkpoint["light_optimizer"]
        # irradiance_volumes_params = checkpoint["irradiance_volumes"]

        gaussians.restore(model_params, opt)
        # cubemap.load_state_dict(cubemap_params)
        # light_optimizer.load_state_dict(light_optimizer_params)
        print(f"Load checkpoint from {checkpoint_path}")

    # define progress bar
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = trange(first_iter, opt.iterations, desc="Training progress")  # For logging

    occlusion_volumes: Dict = {}
    occlusion_flag = True
    occlusion_ids: torch.Tensor
    occlusion_coefficients: torch.Tensor
    occlusion_degree: int
    bound: float
    aabb: torch.Tensor
    for iteration in range(first_iter + 1, opt.iterations + 1):  # the real iteration (1 shift)
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        try:
            c2w = torch.inverse(viewpoint_cam.world_view_transform.T)  # [4, 4]
        except:
            continue

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if iteration <= pbr_iteration:
            background = bg
        else:  # NOTE: black background for PBR
            background = torch.zeros_like(bg)
        rendering_result = render(
            viewpoint_camera=viewpoint_cam,
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            derive_normal=True,
        )
        image = rendering_result["render"]  # [3, H, W]
        viewspace_point_tensor = rendering_result["viewspace_points"]
        visibility_filter = rendering_result["visibility_filter"]
        radii = rendering_result["radii"]
        depth_map = rendering_result["depth_map"]  # [1, H, W]
        normal_map_from_depth = rendering_result["normal_map_from_depth"]  # [3, H, W]
        normal_map = rendering_result["normal_map"]  # [3, H, W]
        albedo_map = rendering_result["albedo_map"]  # [3, H, W]
        roughness_map = rendering_result["roughness_map"]  # [1, H, W]
        metallic_map = rendering_result["metallic_map"]  # [1, H, W]

        # formulate roughness
        rmax, rmin = 1.0, 0.04
        roughness_map = roughness_map * (rmax - rmin) + rmin

        # NOTE: mask normal map by view direction to avoid skip value
        H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
        view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
            .sum(dim=-1)
            .reshape(H, W, 3)
        )  # [H, W, 3]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        gt_image = (gt_image * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)
        loss: torch.Tensor
        Ll1 = F.l1_loss(image, gt_image)
        normal_loss = 0.0
        if iteration <= pbr_iteration:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            # normal loss
            normal_loss_weight = 1.0
            mask = rendering_result["normal_from_depth_mask"]  # [1, H, W]
            normal_loss = F.l1_loss(normal_map[:, mask], normal_map_from_depth[:, mask])
            loss += normal_loss_weight * normal_loss
            normal_tv_loss = get_tv_loss(gt_image, normal_map, pad=1, step=1)
            loss += normal_tv_loss * normal_tv_weight

        else:  # NOTE: PBR
            if occlusion_flag and indirect:
                filepath = os.path.join(os.path.dirname(checkpoint_path), "occlusion_volumes.pth")
                print(f"begin to load occlusion volumes from {filepath}")
                occlusion_volumes = torch.load(filepath)
                occlusion_ids = occlusion_volumes["occlusion_ids"]
                occlusion_coefficients = occlusion_volumes["occlusion_coefficients"]
                occlusion_degree = occlusion_volumes["degree"]
                bound = occlusion_volumes["bound"]
                aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
                occlusion_flag = False
            # recon occlusion
            if indirect:
                points = (
                    (-view_dirs.reshape(-1, 3) * depth_map.reshape(-1, 1) + c2w[:3, 3])
                    .clamp(min=-bound, max=bound)
                    .contiguous()
                )  # [HW, 3]
                occlusion = recon_occlusion(
                    H=H,
                    W=W,
                    bound=bound,
                    points=points,
                    normals=normal_map.permute(1, 2, 0).reshape(-1, 3).contiguous(),
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
                occlusion = torch.ones_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]
                irradiance = torch.zeros_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]

            normal_mask = rendering_result["normal_mask"]  # [1, H, W]
            cubemap.build_mips() # build mip for environment light
            pbr_result = pbr_shading(
                light=cubemap,
                normals=normal_map.permute(1, 2, 0).detach(),  # [H, W, 3]
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
            render_rgb = pbr_result["render_rgb"].permute(2, 0, 1)  # [3, H, W]
            render_rgb = torch.where(
                normal_mask,
                render_rgb,
                background[:, None, None],
            )
            pbr_render_loss = l1_loss(render_rgb, gt_image)
            loss = pbr_render_loss

            ### BRDF loss
            if (normal_mask == 0).sum() > 0:
                brdf_tv_loss = get_masked_tv_loss(
                    normal_mask,
                    gt_image,  # [3, H, W]
                    torch.cat([albedo_map, roughness_map, metallic_map], dim=0),  # [5, H, W]
                )
            else:
                brdf_tv_loss = get_tv_loss(
                    gt_image,  # [3, H, W]
                    torch.cat([albedo_map, roughness_map, metallic_map], dim=0),  # [5, H, W]
                    pad=1,  # FIXME: 8 for scene
                    step=1,
                )
            loss += brdf_tv_loss * brdf_tv_weight
            lamb_weight = 0.001
            lamb_loss = (1.0 - roughness_map[normal_mask]).mean() + metallic_map[normal_mask].mean()
            loss += lamb_loss * lamb_weight

            #### envmap
            # TV smoothness
            envmap = dr.texture(
                cubemap.base[None, ...],
                envmap_dirs[None, ...].contiguous(),
                filter_mode="linear",
                boundary_mode="cube",
            )[
                0
            ]  # [H, W, 3]
            tv_h1 = torch.pow(envmap[1:, :, :] - envmap[:-1, :, :], 2).mean()
            tv_w1 = torch.pow(envmap[:, 1:, :] - envmap[:, :-1, :], 2).mean()
            env_tv_loss = tv_h1 + tv_w1
            loss += env_tv_loss * env_tv_weight

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer=tb_writer,
                iteration=iteration,
                Ll1=Ll1,
                normal_loss=normal_loss,
                loss=loss,
                elapsed=iter_start.elapsed_time(iter_end),
                testing_iterations=testing_iterations,
                scene=scene,
                light=cubemap,
                brdf_lut=brdf_lut,
                canonical_rays=canonical_rays,
                pbr_iteration=pbr_iteration,
                metallic=metallic,
                tone=tone,
                gamma=gamma,
                renderArgs=(pipe, background),
                occlusion_volumes=occlusion_volumes,
                irradiance_volumes=irradiance_volumes,
                indirect=indirect,
            )
            # NOTE: we same .pth instead of point cloud for additional irradiance volumes and cubemap
            # if iteration in saving_iterations:
            #    print(f"\n[ITER {iteration}] Saving Gaussians")
            #    scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.update_learning_rate(iteration)
                if iteration >= pbr_iteration:
                    light_optimizer.step()
                    light_optimizer.zero_grad(set_to_none=True)
                    cubemap.clamp_(min=0.0)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    {
                        "gaussians": gaussians.capture(),
                        "cubemap": cubemap.state_dict(),
                        "irradiance_volumes": irradiance_volumes.state_dict(),
                        "light_optimizer": light_optimizer.state_dict(),
                        "iteration": iteration,
                    },
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )


def prepare_output_and_logger(args: GroupParams) -> Optional[SummaryWriter]:
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer: Optional[SummaryWriter],
    iteration: int,
    Ll1: Union[float, torch.Tensor],
    normal_loss: Union[float, torch.Tensor],
    loss: Union[float, torch.Tensor],
    elapsed: float,
    testing_iterations: List[int],
    scene: Scene,
    light: CubemapLight,
    brdf_lut: torch.Tensor,
    canonical_rays: torch.Tensor,
    pbr_iteration: int,
    metallic: bool,
    tone: bool,
    gamma: bool,
    renderArgs: Tuple[GroupParams, torch.Tensor],
    occlusion_volumes: Dict,
    irradiance_volumes: IrradianceVolumes,
    indirect: bool = False,
) -> None:
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1, iteration)
        tb_writer.add_scalar("train_loss_patches/normal_loss", normal_loss, iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss, iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        if iteration > pbr_iteration and indirect:
            occlusion_ids = occlusion_volumes["occlusion_ids"]
            occlusion_coefficients = occlusion_volumes["occlusion_coefficients"]
            bound = occlusion_volumes["bound"]
            occlusion_degree = occlusion_volumes["degree"]
            aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()

        pipe, background = renderArgs
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    viewpoint: Camera
                    render_result = render(
                        viewpoint_camera=viewpoint,
                        pc=scene.gaussians,
                        pipe=pipe,
                        bg_color=background,
                        inference=True,
                        derive_normal=True,
                    )
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    depth_map = render_result["depth_map"]
                    depth_img = (
                        torch.from_numpy(
                            turbo_cmap(render_result["depth_map"].cpu().numpy().squeeze())
                        )
                        .to(image.device)
                        .permute(2, 0, 1)
                    )
                    normal_map_from_depth = render_result["normal_map_from_depth"]
                    normal_map = render_result["normal_map"]
                    normal_img = torch.cat([normal_map, normal_map_from_depth], dim=-1)
                    gt_image = viewpoint.original_image.cuda()
                    alpha_mask = viewpoint.gt_alpha_mask.cuda()
                    gt_image = (gt_image * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)
                    albedo_map = render_result["albedo_map"]  # [3, H, W]
                    roughness_map = render_result["roughness_map"]  # [1, H, W]
                    metallic_map = render_result["metallic_map"]  # [1, H, W]
                    brdf_map = torch.cat(
                        [
                            albedo_map,
                            torch.tile(roughness_map, (3, 1, 1)),
                            torch.tile(metallic_map, (3, 1, 1)),
                        ],
                        dim=2,
                    )  # [3, H, 3W]
                    # NOTE: PBR record
                    if iteration > pbr_iteration:
                        H, W = viewpoint.image_height, viewpoint.image_width
                        c2w = torch.inverse(viewpoint.world_view_transform.T)  # [4, 4]
                        view_dirs = -(
                            (
                                F.normalize(canonical_rays[:, None, :], p=2, dim=-1)
                                * c2w[None, :3, :3]
                            )  # [HW, 3, 3]
                            .sum(dim=-1)
                            .reshape(H, W, 3)
                        )  # [H, W, 3]
                        normal_mask = render_result["normal_mask"]

                        # recon occlusion
                        if indirect:
                            points = (
                                (-view_dirs.reshape(-1, 3) * depth_map.reshape(-1, 1) + c2w[:3, 3])
                                .clamp(min=-bound, max=bound)
                                .contiguous()
                            )  # [HW, 3]
                            occlusion = recon_occlusion(
                                H=H,
                                W=W,
                                bound=bound,
                                points=points,
                                normals=normal_map.permute(1, 2, 0).reshape(-1, 3).contiguous(),
                                occlusion_coefficients=occlusion_coefficients,
                                occlusion_ids=occlusion_ids,
                                aabb=aabb,
                            ).reshape(H, W, 1)

                            irradiance = irradiance_volumes.query_irradiance(
                                points=points.reshape(-1, 3).contiguous(),
                                normals=normal_map.permute(1, 2, 0).reshape(-1, 3).contiguous(),
                            ).reshape(H, W, -1)
                        else:
                            occlusion = torch.ones_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]
                            irradiance = torch.zeros_like(roughness_map).permute(
                                1, 2, 0
                            )  # [H, W, 1]

                        # build mip for environment light
                        light.build_mips()
                        pbr_result = pbr_shading(
                            light=light,
                            normals=normal_map.permute(1, 2, 0),  # [H, W, 3]
                            view_dirs=view_dirs,
                            mask=normal_mask.permute(1, 2, 0),  # [H, W, 1]
                            albedo=albedo_map.permute(1, 2, 0),  # [H, W, 3]
                            roughness=roughness_map.permute(1, 2, 0),  # [H, W, 1]
                            metallic=metallic_map.permute(1, 2, 0)
                            if metallic
                            else None,  # [H, W, 1]
                            tone=tone,
                            gamma=gamma,
                            brdf_lut=brdf_lut,
                            occlusion=occlusion,
                            irradiance=irradiance,
                        )
                        diffuse_rgb = (
                            pbr_result["diffuse_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
                        )  # [3, H, W]
                        specular_rgb = (
                            pbr_result["specular_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
                        )  # [3, H, W]
                        render_rgb = (
                            pbr_result["render_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
                        )  # [3, H, W]
                        # NOTE: mask render_rgb by depth map
                        background = renderArgs[1]
                        render_rgb = torch.where(
                            normal_mask,
                            render_rgb,
                            background[:, None, None],
                        )
                        diffuse_rgb = torch.where(
                            normal_mask,
                            diffuse_rgb,
                            background[:, None, None],
                        )
                        specular_rgb = torch.where(
                            normal_mask,
                            specular_rgb,
                            background[:, None, None],
                        )
                        pbr_image = torch.cat(
                            [render_rgb, diffuse_rgb, specular_rgb], dim=2
                        )  # [3, H, 3W]
                    else:
                        zero_pad = torch.zeros_like(image)
                        render_rgb = zero_pad
                        pbr_image = torch.cat([zero_pad, zero_pad, zero_pad], dim=2)  # [3, H, 3W]

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}_{idx}/render",
                            resize_tensorboard_img(image)[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}_{idx}/depth",
                            resize_tensorboard_img(depth_img)[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}_{idx}/normal",
                            (resize_tensorboard_img(normal_img, 1600)[None] + 1.0) / 2.0,
                            global_step=iteration,
                        )
                        if iteration > pbr_iteration:
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}_{idx}/brdf",
                                resize_tensorboard_img(brdf_map, 2400)[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}_{idx}/pbr_render",
                                resize_tensorboard_img(pbr_image, 2400)[None],
                                global_step=iteration,
                            )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}_{idx}/ground_truth",
                                resize_tensorboard_img(gt_image)[None],
                                global_step=iteration,
                            )
                    if iteration > pbr_iteration:
                        l1_test += F.l1_loss(render_rgb, gt_image).mean().double()
                        psnr_test += psnr(render_rgb, gt_image).mean().double()
                        ssim_test += ssim(render_rgb, gt_image).mean().double()
                    else:
                        l1_test += F.l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        ssim_test += ssim(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                ssim_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.6f} PSNR {psnr_test:.6f} SSIM {ssim_test:.6f}"
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity.reshape(-1), iteration
            )
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[7_000, 30_000, 37_000],
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[7_000, 30_000, 37_000],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default=None, help="The path to the checkpoint to load.")
    parser.add_argument("--pbr_iteration", default=30_000, type=int, help="The iteration to begin the pb.r learning (Deomposition Stage in the paper)")
    parser.add_argument("--normal_tv", default=5.0, type=float, help="The weight of TV loss on predicted normal map.")
    parser.add_argument("--brdf_tv", default=1.0, type=float, help="The weight of TV loss on predicted BRDF (material) map.")
    parser.add_argument("--env_tv", default=0.01, type=float, help="The weight of TV loss on Environment Map.")
    parser.add_argument("--bound", default=1.5, type=float, help="The valid bound of occlusion volumes.")
    parser.add_argument("--tone", action="store_true", help="Enable aces film tone mapping.")
    parser.add_argument("--gamma", action="store_true", help="Enable linear_to_sRGB for gamma correction.")
    parser.add_argument("--metallic", action="store_true", help="Enable metallic material reconstruction.")
    parser.add_argument("--indirect", action="store_true", help="Enable indirect diffuse modeling.")
    args = parser.parse_args(sys.argv[1:])
    args.test_iterations.append(args.iterations)
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint_iterations=args.checkpoint_iterations,
        checkpoint_path=args.start_checkpoint,
        pbr_iteration=args.pbr_iteration,
        debug_from=args.debug_from,
        metallic=args.metallic,
        tone=args.tone,
        gamma=args.gamma,
        normal_tv_weight=args.normal_tv,
        brdf_tv_weight=args.brdf_tv,
        env_tv_weight=args.env_tv,
        bound=args.bound,
        indirect=args.indirect,
    )

    # All done
    print("\nTraining complete.")
