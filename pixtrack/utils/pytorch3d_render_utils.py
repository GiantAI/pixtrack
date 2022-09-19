import torch
import trimesh
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as R
from hloc.utils.read_write_model import Camera, Image
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    look_at_rotation,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointLights, 
    AmbientLights,
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex)


def create_look_at_camera_poses(radius, subdivisions=2):
    sph = trimesh.creation.icosphere(subdivisions=subdivisions, 
                                     radius=radius, 
                                     color=None)
    v = sph.vertices
    Rs = [look_at_rotation(torch.tensor(v[i]).unsqueeze(0).float()).squeeze().numpy() for i in range(v.shape[0])]
    Rs = [R.from_matrix(x).as_matrix() for x in Rs]
    Ts = [-Rs[i].T @ v[i] for i in range(v.shape[0])]
    Rs = torch.tensor(np.stack(Rs))
    Ts = torch.tensor(np.stack(Ts))
    return Rs, Ts

def create_look_at_poses_for_mesh(mesh_path, subdivisions=2):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    mesh = load_objs_as_meshes([mesh_path], device=device)
    torch.min(mesh.verts_list()[0], dim=0).values
    mesh_min = torch.min(mesh.verts_list()[0], dim=0).values
    mesh_max = torch.max(mesh.verts_list()[0], dim=0).values
    max_dist = torch.sqrt(torch.sum((mesh_max - mesh_min) ** 2))
    max_dist = float(max_dist)
    Rs, Ts = create_look_at_camera_poses(max_dist, subdivisions)
    return Rs, Ts, mesh

def render_image(mesh, fx, fy, cx, cy, W, H, R, T, device='cuda:0'):

    # Create cameras
    assert fx == fy
    focal_length = fx
    principal_point = (cx, cy)
    image_size = (W, H)
    cameras = PerspectiveCameras(focal_length=focal_length,
                             principal_point=(principal_point,),
                             device=device, 
                             R=R, 
                             T=T,
                             in_ndc=False,
                             image_size=(image_size,))

    # Rasterization Settings
    raster_settings = RasterizationSettings(
                             image_size=image_size, 
                             blur_radius=0.0, 
                             faces_per_pixel=1,)

    # Create lights
    lights = AmbientLights(device=device)

    # Create renderer
    renderer = MeshRenderer(
                            rasterizer=MeshRasterizer(
                                 cameras=cameras, 
                                 raster_settings=raster_settings
                            ),
                            shader=SoftPhongShader(
                                 device=device, 
                                 cameras=cameras,
                                 lights=lights
                            )
                            )

    # Render image
    image = renderer(mesh).squeeze().cpu().numpy()

    return image

def create_colmap_image_from_pytorch3d_RT(R, T, image_name, image_id, camera_id):
    
    # Flip x, y axes
    flip = np.eye(3)
    flip[0, 0] = -1.
    flip[1, 1] = -1.
    R_colmap = R @ flip

    # Create colmap image
    qvec = pycolmap.rotmat_to_qvec(R_colmap.T)
    tvec = T
    image = Image(id=image_id, qvec=qvec, tvec=tvec,
                  camera_id=camera_id, name=image_name,
                  xys=[], point3D_ids=[])
    return image

def create_colmap_camera(w, h, f, cx, cy, k1, camera_id=1, model_id='SIMPLE_RADIAL'):
    assert model_id == 'SIMPLE_RADIAL'
    params = tuple([f, cx, cy, k1])
    cam = Camera(id=camera_id,
                 model=model_id,
                 width=w,
                 height=h,
                 params = np.array(params))
    return cam

