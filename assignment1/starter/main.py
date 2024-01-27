from utils import *
import pytorch3d
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw

import utils as utils
import dolly_zoom as dolly_zoom
from camera_transforms import render_cow
from render_generic import *

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))

def render(R, T, model=([], []), image_size=256, color=[0.7, 0.7, 1], device=None, textures=None):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = model
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    texture = torch.ones_like(vertices)  # (1, N_v, 3)
    if textures is None:
        texture = texture * torch.tensor(color)  # (1, N_v, 3)
    else:
        texture = textures
        texture = texture.unsqueeze(0) 

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(texture),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 3, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend

def render360(filepath, camera_dist = 3, elevation = 30, azimuth = 360, model="data/cow.obj", image_size=256, textures=None):
    img_list = []

    print("Rendering a 360 degree view of the model ....")
    if type(model) == str:
        vertices, faces = load_cow_mesh(model)
    else:
        vertices, faces = model

    for i in tqdm(range(0, azimuth+1, 10)):
        R, T = pytorch3d.renderer.look_at_view_transform(dist = camera_dist, elev = elevation, azim = i)
        img = render(R, T, model=(vertices, faces), image_size=image_size, textures=textures)
        img = Image.fromarray((img * 255).astype(np.uint8))
        img_list.append(img)

    # print(len(img_list))
    imageio.mimsave(filepath, img_list, fps=10)

def generateTetrahedron():
    vertices = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 0, 0.866],
        [0.5, 0.816, 0.289]
    ])
    faces = torch.tensor([
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3]
    ])
    return vertices, faces

def generateCube():
    vertices = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ])
    # Coinvert tensor to float
    vertices = vertices.float()
    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 5, 1],
        [0, 4, 5],
        [1, 6, 2],
        [1, 5, 6],
        [2, 7, 3],
        [2, 6, 7],
        [3, 4, 0],
        [3, 7, 4]
    ])
    return vertices, faces

def retexturing(model=([],[]), color1 = [0, 0, 1], color2 = [1, 0, 0]):
    vertices, faces = model
    textures = torch.ones_like(vertices)
    z_min = torch.min(vertices[:, 2])
    z_max = torch.max(vertices[:, 2])
    
    for i in range(len(vertices)):
        alpha = (vertices[i][2] - z_min) / (z_max - z_min)
        color = alpha * torch.tensor(color2) + (1 - alpha) * torch.tensor(color1)
        textures[i] = color

    return textures

def render_pointcloud(points = None, colors = None, R = None, T = None,
    image_size=256, background_color=(1, 1, 1), device=None):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(image_size=image_size, background_color=background_color)
    point_cloud = pytorch3d.structures.Pointclouds(points=points, features=colors)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def renderpointcloud_360(points = [], colors = [], filepath = "results/pointcloud.gif", image_size = 256):
    img_list = []
    print("Rendering a 360 degree view of the model ....")
    for i in tqdm(range(0, 360+1, 10)):
        R, T = pytorch3d.renderer.look_at_view_transform(dist = 6, elev = 30, azim = i)
        img = render_pointcloud(R, T, points, colors, image_size)
        img = Image.fromarray((img * 255).astype(np.uint8))
        img_list.append(img)

    # print(len(img_list))
    imageio.mimsave(filepath, img_list, fps=10)

if __name__ == "__main__":
    # # Question 1.1
    # print("Solution for Question 1.1 ...")
    # render360('results/cow_360.gif')

    # # Question 1.2
    # print("Solution for Question 1.2 ...")
    # dolly_zoom.dolly_zoom(output_file="results/dolly_zoom.gif")

    # # Question 2.1
    # print("Solution for Question 2.1 ...")
    # model = generateTetrahedron()
    # render360('results/tetrahedron_360.gif', model=model)

    # # Question 2.2
    # print("Solution for Question 2.2 ...")
    # model = generateCube()
    # render360('results/cube_360.gif', model=model)

    # Question 3
    # model = load_cow_mesh("data/cow.obj")
    # textures = retexturing(model=model)
    # render360('results/cow_360_retextured.gif', model=model, textures=textures)

    # # Question 4
    # # Image 1 - Rotate camera anti-clockwise 90 degrees along the Z axis
    # R1 = torch.tensor([
    #     [0, -1, 0],
    #     [1, 0, 0],
    #     [0, 0, 1]
    # ])
    # T1 = torch.tensor([0, 0, 0]) 
    # plt.imsave("results/Q4_img1.jpg", render_cow(R_relative=R1, T_relative=T1))

    # # Image 2 - Zoom out by a factor of 2
    # R2 = torch.tensor([
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ])
    # T2 = torch.tensor([0, 0, 3])
    # plt.imsave("results/Q4_img2.jpg", render_cow(R_relative=R2, T_relative=T2))

    # # Image 3 - Translate the camera by 0.5 unit along the X axis and Y axis
    # R3 = torch.tensor([
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ])
    # T3 = torch.tensor([0.5, -0.5, 0])
    # plt.imsave("results/Q4_img3.jpg", render_cow(R_relative=R3, T_relative=T3))

    # # Image 4 - Rotate camera clockwise 90 degrees along the Y axis of the cow
    # R4 = torch.tensor([
    #     [0, 0, -1],
    #     [0, 1, 0],
    #     [1, 0, 0]
    # ])
    # T4 = torch.tensor([3, 0, 3])
    # plt.imsave("results/Q4_img4.jpg", render_cow(R_relative=R4, T_relative=T4))

    # Question 5
    data = load_rgbd_data()
    # Split the data into respective images
    data1 = {}
    data2 = {}
    for key in data:
        if key[-1]  == '1':
            data1[key[:-1]] = data[key]
        else:
            data2[key[:-1]] = data[key]
    
    point1, color1 = unproject_depth_image(torch.tensor(data1['rgb']), torch.tensor(data1['mask']), torch.tensor(data1['depth']), data1['cameras'])
    point2, color2 = unproject_depth_image(torch.tensor(data2['rgb']), torch.tensor(data2['mask']), torch.tensor(data2['depth']), data2['cameras'])

    # drop the aplha channels from the color tensors
    color1 = color1[:, :3]
    color2 = color2[:, :3]

    point1 = point1.unsqueeze(0)
    point2 = point2.unsqueeze(0)
    color1 = color1.unsqueeze(0)
    color2 = color2.unsqueeze(0)

    print(point1.shape)
    print(color1.shape)
    print(point2.shape)
    print(color2.shape)

    renderpointcloud_360(points = point1, colors = color1, filepath = "results/pointcloud1.gif")
    renderpointcloud_360(points = point2, colors = color2, filepath = "results/pointcloud2.gif")





