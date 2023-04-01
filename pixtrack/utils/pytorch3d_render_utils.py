import math
import torch
import trimesh
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as R
from hloc.utils.read_write_model import Camera, Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_rotation,
    PerspectiveCameras,
    AmbientLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams,
)


def create_look_at_camera_poses(radius, subdivisions=2):
    sph = trimesh.creation.icosphere(
        subdivisions=subdivisions, radius=radius, color=None
    )
    v = sph.vertices
    Rs = [
        look_at_rotation(torch.tensor(v[i]).unsqueeze(0).float()).squeeze().numpy()
        for i in range(v.shape[0])
    ]
    Rs = [R.from_matrix(x).as_matrix() for x in Rs]
    Ts = [-Rs[i].T @ v[i] for i in range(v.shape[0])]
    Rs = torch.tensor(np.stack(Rs))
    Ts = torch.tensor(np.stack(Ts))
    return Rs, Ts


def create_look_at_poses_for_mesh(
    fx, fy, W, H, mesh_path, subdivisions=2, device=torch.device("cuda:0")
):
    mesh = load_objs_as_meshes([mesh_path], device=device)
    torch.min(mesh.verts_list()[0], dim=0).values
    mesh_min = torch.min(mesh.verts_list()[0], dim=0).values
    mesh_max = torch.max(mesh.verts_list()[0], dim=0).values
    max_dist = torch.sqrt(torch.sum((mesh_max - mesh_min) ** 2))
    radius = float(max_dist) / 2.

    angle_x = math.atan(W / (fx * 2))
    angle_y = math.atan(H / (fy * 2))
    d = max(radius / math.sin(angle_x), radius / math.sin(angle_y))

    Rs, Ts = create_look_at_camera_poses(d, subdivisions)
    return Rs, Ts, mesh


def render_image(mesh, fx, fy, cx, cy, W, H, R, T, device="cuda:0"):

    # Create cameras
    assert fx == fy
    focal_length = fx
    principal_point = (cx, cy)
    image_size = (H, W)
    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=(principal_point,),
        device=device,
        R=R,
        T=T,
        in_ndc=False,
        image_size=(image_size,),
    )

    # Rasterization Settings
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Create lights
    lights = AmbientLights(device=device)

    # Create renderer
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend_params),

    )

    # Render image
    image = renderer(mesh).squeeze().cpu().numpy()

    return image


def create_colmap_image_from_pytorch3d_RT(R, T, image_name, image_id, camera_id):

    # Flip x, y axes
    flip = np.eye(3)
    flip[0, 0] = -1.0
    flip[1, 1] = -1.0
    R_colmap = R @ flip

    # Create colmap image
    qvec = pycolmap.rotmat_to_qvec(R_colmap.T)
    tvec = T
    image = Image(
        id=image_id,
        qvec=qvec,
        tvec=tvec,
        camera_id=camera_id,
        name=image_name,
        xys=[],
        point3D_ids=[],
    )
    return image


def create_colmap_camera(w, h, f, cx, cy, k1, camera_id=1, model_id="SIMPLE_RADIAL"):
    assert model_id == "SIMPLE_RADIAL"
    params = tuple([f, cx, cy, k1])
    cam = Camera(
        id=camera_id,
        model=model_id,
        width=w,
        height=h,
        params=np.array(params),
    )
    return cam
