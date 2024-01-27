from utils import *
import pytorch3d
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw

import utils as utils
import dolly_zoom as dolly_zoom

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
    model = load_cow_mesh("data/cow.obj")
    textures = retexturing(model=model)
    render360('results/cow_360_retextured.gif', model=model, textures=textures)



