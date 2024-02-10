import numpy as np
import matplotlib.pyplot as plt
import pytorch3d
import torch
from utils import get_device, get_mesh_renderer, get_points_renderer
from tqdm.auto import tqdm
import imageio
# from PIL import Image, ImageDraw

"""
Functions to visualize result mesh, voxel, point clouds
"""

def visualize_mesh(mesh,
                   textures=None, 
                   output_path='meshviz.gif',
                   distance=1.0,
                   fov=60,
                   image_size=256,
                   color=[0.7, 0.7, 1],
                   steps=range(360, 0, -15)):
    device = get_device()

    vertices, faces = mesh.verts_list()[0], mesh.faces_list()[0]
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    if textures is None:
        # Get the vertices, faces, and textures.
        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        textures = (vertices - vertices.min()) / \
                   (vertices.max() - vertices.min())

    render_mesh = pytorch3d.structures.Meshes(
    verts=vertices,
    faces=faces,
    textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    render_mesh = render_mesh.to(device)       

    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, 3.0]], device=device)

    images = []
    for i in tqdm(steps):

        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=distance, 
                                                                 elev=0.5, 
                                                                 azim=i, at=((0, 0, 0), ), 
                                                                 device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=T,  
                                                           R=R, 
                                                           fov=fov, 
                                                           device=device)
        rend = renderer(render_mesh, cameras=cameras, lights=lights)
        image = (rend*255).detach().cpu().numpy().astype(np.uint8).squeeze(0)
        images.append(image)

    imageio.mimsave(output_path, images, duration=0.3, format='gif', disposal=2, loop=0)
    return

def visualize_voxel(voxels,
                   textures=None, 
                   output_path='voxelviz.gif',
                   distance=3.0,
                   fov=60,
                   image_size=256,
                   color=[0.7, 0.7, 1],
                   steps=range(360, 0, -15)):
    device = get_device()
    mesh = pytorch3d.ops.cubify(voxels, thresh=0.8).to(device)
    visualize_mesh(mesh, 
                   textures=textures, 
                   output_path=output_path,
                   distance=distance,
                   fov=fov,
                   image_size=image_size,
                   color=color,
                   steps=steps)
    return

def visualize_pcd(point_cloud_src,
                rgb=None,
                output_path='pcdviz.gif',
                distance=1.0,
                fov=60,
                image_size=256,
                background_color=[1., 1, 1],
                steps=range(360, 0, -15)):
    device = get_device()
    points = point_cloud_src
    if rgb==None:
      rgb = (points - points.min()) / (points.max() - points.min())

    render_point_cloud = pytorch3d.structures.Pointclouds(
      points=[points], features=[rgb],
    ).to(device)

    renderer = get_points_renderer(image_size=image_size,
                                    device=device) #,
                                    # background_color=background_color)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]],
                                            device=device)

    images = []
    for i in tqdm(steps):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=distance,
                                                                # elev=0.5, 
                                                                azim=i, 
                                                                at=((0, 0, 0), ))
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=T,  R=R, device=device)
        rend = renderer(render_point_cloud, cameras=cameras)

        image = (rend.detach().cpu().numpy()[0, ..., :3] * 255).astype(np.uint8)
        images.append(image)
    
    imageio.mimsave(output_path, images, duration=0.3, format='gif', disposal=2, loop=0)
    return