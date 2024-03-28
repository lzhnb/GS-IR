import os
import glob
from argparse import ArgumentParser

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_mae(gt_normal_stack: np.ndarray, render_normal_stack: np.ndarray) -> float:
    # compute mean angular error
    MAE = np.mean(
        np.arccos(np.clip(np.sum(gt_normal_stack * render_normal_stack, axis=-1), -1, 1))
        * 180
        / np.pi
    )
    return MAE.item()


if __name__ == "__main__":
    parser = ArgumentParser(description="TensoIR convert script parameters")
    parser.add_argument("--output_dir", type=str, help="The path to the output directory that stores the predicted normal results.")
    parser.add_argument("--gt_dir", type=str, help="The path to the output directory that stores the normal ground truth.")
    args = parser.parse_args()

    output_dir = args.output_dir

    test_dirs = glob.glob(os.path.join(args.gt_dir, "test_*"))
    test_dirs.sort()

    normal_gt_stack = []
    normal_gs_stack = []
    normal_from_depth_stack = []

    normal_bg = np.array([0.0, 0.0, 1.0])
    for test_dir in tqdm(test_dirs):
        test_id = int(test_dir.split("_")[-1])
        normal_gt_path = os.path.join(test_dir, "normal.png")
        normal_gt_img = Image.open(normal_gt_path)
        normal_gt = np.array(normal_gt_img)[..., :3] / 255  # [H, W, 3] in range [0, 1]
        normal_gt = (normal_gt - 0.5) * 2.0  # [H, W, 3] in range (-1, 1)
        alpha_mask = np.array(normal_gt_img)[..., [-1]] / 255  # [H, W, 1] in range [0, 1]
        normal_gt = normal_gt * alpha_mask + normal_bg * (1.0 - alpha_mask)  # [H, W, 3]
        normal_gt = normal_gt / np.linalg.norm(normal_gt, axis=-1, ord=2, keepdims=True)
        normal_gt_stack.append(normal_gt)

        # gs normal
        normal_gs_path = os.path.join(output_dir, "normal", f"{test_id:05d}_normal.png")
        normal_gs_img = Image.open(normal_gs_path)
        normal_gs = np.array(normal_gs_img)[..., :3] / 255  # [H, W, 3] in range [0, 1]
        normal_gs = (normal_gs - 0.5) * 2.0  # [H, W, 3] in range (-1, 1)
        # NOTE: a trick to tackle (128 / 255)
        mask = (np.array(normal_gs_img)[..., :3] == np.array([128, 128, 255], dtype=np.uint8)).all(-1)
        normal_gs[mask] = np.array([0.0, 0.0, 1.0])
        normal_gs = normal_gs / np.linalg.norm(normal_gs, axis=-1, ord=2, keepdims=True)
        normal_gs_stack.append(normal_gs)

        # normal from depth
        normal_from_depth_path = os.path.join(output_dir, "normal", f"{test_id:05d}_from_depth.png")
        normal_from_depth_img = Image.open(normal_from_depth_path)
        normal_from_depth = (
            np.array(normal_from_depth_img)[..., :3] / 255
        )  # [H, W, 3] in range [0, 1]
        normal_from_depth = (normal_from_depth - 0.5) * 2.0  # [H, W, 3] in range (-1, 1)
        # mask_from_depth = (normal_from_depth == 0).all(-1)
        mask = (np.array(normal_from_depth_img)[..., :3] == np.array([128, 128, 255], dtype=np.uint8)).all(-1)
        normal_from_depth[mask] = np.array([0.0, 0.0, 1.0])
        normal_from_depth = normal_from_depth / np.linalg.norm(
            normal_from_depth, axis=-1, ord=2, keepdims=True
        )
        normal_from_depth_stack.append(normal_from_depth)

    # MAE
    normal_gt_stack = np.stack(normal_gt_stack)
    normal_gs_stack = np.stack(normal_gs_stack)
    normal_from_depth_stack = np.stack(normal_from_depth_stack)
    mae_gs = get_mae(normal_gt_stack, normal_gs_stack)
    mae_from_depth = get_mae(normal_gt_stack, normal_from_depth_stack)
    print(f"MAE: gs={mae_gs}; from_depth={mae_from_depth}")

