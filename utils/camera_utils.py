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

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d
from tqdm import tqdm

from arguments import GroupParams
from scene.cameras import Camera
from scene.dataset_readers import CameraInfo
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False


def loadCam(args: GroupParams, id: int, cam_info: CameraInfo, resolution_scale: float) -> Camera:
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
    )


def cameraList_from_camInfos(
    cam_infos: List[CameraInfo], resolution_scale: float, args: GroupParams
) -> List[Camera]:
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos)):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id: int, camera: Camera) -> Dict:
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry


def get_camera_mesh(
    extrinsic: Union[torch.Tensor, np.ndarray],
    intrinsic: Optional[Union[torch.Tensor, np.ndarray]] = None,
    is_c2w: bool = False,
    HW: Tuple[int] = None,
    camera_size: float = 1.0,
    camera_color: Sequence[float] = [1.0, 1.0, 1.0],
    up_color: Sequence[float] = [1.0, 0.0, 0.0],
    right_color: Sequence[float] = [0.0, 1.0, 0.0],
) -> trimesh.Trimesh:
    """get the mesh of the given camera pose

    Args:
        extrinsic (Union[torch.Tensor, np.ndarray]): extinsics of camera for visualization. The shape should be [3/4, 4]
        intrinsic (Optional[Union[torch.Tensor, np.ndarray]], optional): intrinsic of camera for visualization.
            the shape should be [3, 3]. Defaults to None.
        is_c2w (bool, optional): whether the input extrnsic is the matrix that transform camera coordinate to
            world coordinate. Defaults to False.
        HW (Tuple[int], optional): image resolution. If not specified, given (cy * 2, cx * 2). Defaults to None.
        camera_size (float, optional): control the scale of the camera mesh. Defaults to 1.0.
        up_color (Sequence[float], optional): the color to shade up indication faces. Defaults to [1, 0, 0].
        right_color (Sequence[float], optional): the color to shade right indication faces. Defaults to [0, 1, 0].

    Returns:
        trimesh.Trimesh: camera mesh
    """

    """format the extrinsic and intrinsic"""
    # fmt: off
    default_intrinsic = np.array([
        [1000.0,    0.0, 500.0],
        [   0.0, 1000.0, 500.0],
        [   0.0,    0.0,   1.0]
    ]).astype(np.float32)
    # fmt: on
    extrinsic = np.asarray(extrinsic).astype(np.float32)
    intrinsic = (
        np.asarray(intrinsic).astype(np.float32) if intrinsic is not None else default_intrinsic
    )
    # dimension assert
    assert (
        extrinsic.ndim == 2 and extrinsic.shape[0] in [3, 4] and extrinsic.shape[1] == 4
    ), f"extrinsic should be of shape [3/4, 4], but got {extrinsic.shape}"
    if extrinsic.shape[0] == 3:
        extrinsic = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], axis=0)
    assert (
        intrinsic.ndim == 2 and intrinsic.shape[0] == 3 and intrinsic.shape[1] == 3
    ), f"intrinsic should be of shape [3, 3], but got {intrinsic.shape}"

    # inverse the extrinsic
    pose = np.linalg.inv(extrinsic) if not is_c2w else extrinsic

    # the canonical camera
    camera_faces = np.array(
        [
            # frustum
            [0, 1, 2],
            [0, 2, 4],
            [0, 4, 3],
            [0, 3, 1],
            # up
            [5, 7, 6],  # NOTE: reorder for normal shading
            # right
            [8, 10, 9],  # NOTE: reorder for normal shading
        ]
    ).astype(np.int32)
    camera_colors = np.ones_like(camera_faces).astype(np.float32)
    # shading camera
    camera_colors[:-2, :] = np.asarray(camera_color, dtype=np.float32)
    # shading up and right indication faces
    camera_colors[-2] = np.asarray(up_color, dtype=np.float32)
    camera_colors[-1] = np.asarray(right_color, dtype=np.float32)

    """get the vertices of cameras mesh"""
    H, W = HW if HW is not None else (intrinsic[1, 2] * 2, intrinsic[0, 2] * 2)
    # use to div the cx/cy by fx/fy and get the frustum vertices
    focal_div = np.linalg.inv(intrinsic[:2, :2])
    # fmt: off
    tl = (focal_div @ - intrinsic[:2, 2:3]).reshape(-1) # top left
    tr = (focal_div @ (np.array([[W,], [0,]]) - intrinsic[:2, 2:3])).reshape(-1) # top right
    bl = (focal_div @ (np.array([[0,], [H,]]) - intrinsic[:2, 2:3])).reshape(-1) # bottom left
    br = (focal_div @ (np.array([[W,], [H,]]) - intrinsic[:2, 2:3])).reshape(-1) # bottom right

    camera_vertices = np.array([
        [0.0,      0.0,  0.0],
        [tl[0],  tl[1],  1.0],  # tl
        [tr[0],  tr[1],  1.0],  # tr
        [bl[0],  bl[1],  1.0],  # bl
        [br[0],  br[1],  1.0],  # br

        # up
        [tl[0] * 0.8, tl[1] * 1.1, 1.0],
        [tr[0] * 0.8, tr[1] * 1.1, 1.0],
        [0.0,         tl[1] * 1.5, 1.0],

        # right
        [tr[0] * 1.05, tr[1] * 0.4, 1.0],
        [br[0] * 1.05, br[1] * 0.4, 1.0],
        [tr[0] * 1.2,  0.0,         1.0]
    ], dtype=np.float32) * camera_size
    # fmt: on

    """tranform the camera vertices and get the final camera mesh"""
    camera_vertices = np.pad(camera_vertices, ((0, 0), (0, 1)), "constant", constant_values=(1, 1))
    camera_vertices = camera_vertices @ pose.T
    camera_vertices = camera_vertices[:, :3] / camera_vertices[:, 3:]

    mesh = trimesh.Trimesh(vertices=camera_vertices, faces=camera_faces, face_colors=camera_colors)

    return mesh


def save_camera_mesh(
    extrinsics: Union[torch.Tensor, np.ndarray],
    intrinsics: Optional[Union[torch.Tensor, np.ndarray]] = None,
    is_c2w: bool = True,
    camera_size: Union[float, str] = "auto",
    HW: Tuple[int] = None,
    camera_color: Sequence[float] = [1.0, 1.0, 1.0],
    up_color: Sequence[float] = [1.0, 0.0, 0.0],
    right_color: Sequence[float] = [0.0, 1.0, 0.0],
    verbose: bool = True,
    path: str = "./cameras.ply",
) -> None:
    """modify from MaLi's function, Save camera mesh

    Args:
        extrinsics (Union[torch.Tensor, np.ndarray]):
            extinsics of cameras for visualization, the shape can be [3/4, 4] or [num_cameras, 3/4, 4].
        intrinsics (Optional[Union[torch.Tensor, np.ndarray]], optional):
            intrinsic of cameras for visualization, the shape can be [3, 3] or [num_cameras, 3, 3]. Defaults to None.
        is_c2w (bool, optional): whether the input extrnsic is the matrix that transform camera coordinate to
            world coordinate. Defaults to True.
        camera_size (Union[float, str], optional): control the scale of the camera mesh, when "auto",
            the size is determined by the radius of camera origins. Defaults to "auto".
        HW (Tuple[int], optional): image resolution. If not specified, given (cy * 2, cx * 2). Defaults to None.
        up_color (Sequence[float], optional): the color to shade up indication faces. Defaults to [1, 0, 0] (Red).
        right_color (Sequence[float], optional): the color to shade right indication faces. Defaults to [0, 1, 0] (Green).
        verbose (bool, optional): whether print export path. Defaults to True.
        path (str): path of the export trimesh. Defaults to "./cameras.ply"

    Raises:
        ValueError: invalid extrinsics of intrinsics dimensions.
    """
    if extrinsics.ndim == 2:
        """just one camera"""
        camera_size = 0.1 if camera_size == "auto" else float(camera_size)
        camera_mesh = get_camera_mesh(
            extrinsics,
            intrinsics,
            is_c2w=is_c2w,
            camera_size=camera_size,
            HW=HW,
            camera_color=camera_color,
            up_color=up_color,
            right_color=right_color,
        )
        camera_mesh.export(path)
    elif extrinsics.ndim == 3:
        """multiple cameras"""
        num_cameras = extrinsics.shape[0]
        if intrinsics is not None:
            if intrinsics.ndim == 2:
                intrinsics = [intrinsics] * num_cameras
            else:
                assert intrinsics.ndim == 3 and intrinsics.shape[0] == extrinsics.shape[0]
        if camera_size == "auto":
            # calculate the camera size automatically by the radius of camera origins
            origins = np.asarray(extrinsics[:, :3, 3])
            center = np.mean(origins, axis=0)
            radius = np.linalg.norm((origins - center).max(0))
            camera_size = radius * 0.05
        else:
            camera_size = float(camera_size)

        all_camera_meshes = []
        # get all camera meshes and concatenate them together
        for cam_id in range(num_cameras):
            # get the extrinsic and intrinsic
            extrinsic = extrinsics[cam_id]
            intrinsic = None if intrinsics is None else intrinsics[cam_id]
            # get the shading colors of camera/up/right
            # shading all camera by one color or specified each camera's color
            cam_color = (
                camera_color
                if not isinstance(camera_color[0], (Sequence, np.ndarray))
                else camera_color[cam_id]
            )
            u_color = (
                up_color
                if not isinstance(up_color[0], (Sequence, np.ndarray))
                else up_color[cam_id]
            )
            r_color = (
                right_color
                if not isinstance(right_color[0], (Sequence, np.ndarray))
                else right_color[cam_id]
            )
            camera_mesh = get_camera_mesh(
                extrinsic,
                intrinsic,
                is_c2w=is_c2w,
                camera_size=camera_size,
                HW=HW,
                camera_color=cam_color,
                up_color=u_color,
                right_color=r_color,
            )
            all_camera_meshes.append(camera_mesh)
        all_camera_mesh = trimesh.util.concatenate(all_camera_meshes)
        all_camera_mesh.export(path)
    else:
        raise ValueError(f"the dimensino of extrinsics must be 2 or 3, but got {extrinsics.ndim}")

    if verbose:
        print(f"export the camera mesh at {path}")


def trajectory_from_c2ws(c2ws: List[np.ndarray], frames: int) -> List[np.ndarray]:
    """generate trajector from given c2ws

    Args:
        c2ws (List[np.ndarray]): list of c2ws
        frames (int): the number of output frames

    Returns:
        List[np.ndarray]: the interpolated c2ws of trajectory from given c2ws
    """
    # store key frames and rotation for slerp
    rots = []
    key_times = []
    pos_x = []
    pos_y = []
    pos_z = []
    for key_id, c2w in enumerate(c2ws):
        pos_x.append(c2w[0, 3])
        pos_y.append(c2w[1, 3])
        pos_z.append(c2w[2, 3])
        rots.append(c2w[:3, :3])
        key_times.append(key_id)
    key_rots = R.from_matrix(np.stack(rots))
    slerp = Slerp(key_times, key_rots)
    lerp_x = interp1d(key_times, np.array(pos_x), "cubic")
    lerp_y = interp1d(key_times, np.array(pos_y), "cubic")
    lerp_z = interp1d(key_times, np.array(pos_z), "cubic")

    # get the times for interpolation
    times = []
    for i in range(frames):
        curr = i / frames * (len(c2ws) - 1)
        times.append(curr)
    
    # interpolation generation
    rots_inter = slerp(times).as_matrix()
    x_inter = lerp_x(times)
    y_inter = lerp_y(times)
    z_inter = lerp_z(times)

    # pose
    c2ws_inter = []
    for i in range(frames):
        c2w_inter = np.eye(4)
        c2w_inter[:3, :3] = rots_inter[i]
        c2w_inter[:3, 3] = np.array([x_inter[i], y_inter[i], z_inter[i]])
        c2ws_inter.append(c2w_inter)

    return c2ws_inter
