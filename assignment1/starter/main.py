from utils import *
import pytorch3d
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
import pickle 
import mcubes
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)


import utils as utils
import dolly_zoom as dolly_zoom
from camera_transforms import render_cow
# from render_generic import *

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))

def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

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
    point_cloud = pytorch3d.structures.Pointclouds(points=points.unsqueeze(0), features=colors.unsqueeze(0))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def renderpointcloud_360(points = [], colors = [], filepath = "results/pointcloud.gif", image_size = 256, flip = False):
    img_list = []
    print("Rendering a 360 degree view of the model ....")
    for i in tqdm(range(0, 360+1, 10)):
        R, T = pytorch3d.renderer.look_at_view_transform(dist = 7, elev = 30, azim = i)
        if flip:
            # Rotate the R about the Z axis by 180 degrees
            R = torch.tensor([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ]).float() @ R
            

        img = render_pointcloud(R = R, T = T, points = points, colors = colors, image_size = image_size)
        img = Image.fromarray((img * 255).astype(np.uint8))
        # img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_list.append(img)

    # print(len(img_list))
    imageio.mimsave(filepath, img_list, fps=10)

# Function that performs sampling on tghe parametric function of a torus. Sample number is passed as an input and the torus pointcloud returned as the output
def generateTorus(num_samples = 100):
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (2 + torch.cos(Theta)) * torch.cos(Phi)
    y = (2 + torch.cos(Theta)) * torch.sin(Phi)
    z = torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    return points, color

def generateTorusMesh(image_size=256, voxel_size=64):
    min_value = -3.1
    max_value = 3.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = (X**2 + Y**2 + Z**2 + 4 - 1)**2 - 16 * (X**2 + Y**2)
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures)

    return mesh

def findVisibilityMatrix(mesh = None, camera = None, image_size = 512, render = False):
    if mesh == None or camera == None:
        print("One or more required arguements missing\nAborting...")
        return -1
    # Extract vertices and faces from the mesh
    vertices = mesh.verts_packed().cpu().numpy()
    faces = mesh.faces_packed().cpu().numpy()

    # Set up the rasterizer
    raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

    # Extract the raster data. We do not need to forward it to the renderer
    fragments = rasterizer.forward(mesh)

    # Extracy visible mesh elements using the pix_to_face attribute
    N = 0
    k = 0
    face_list = list(
                    sorted(
                        set(
                            [int(fragments.pix_to_face[N, y, x, k])
                                for x in range(0, image_size)
                                for y in range(0, image_size)
                                if int(fragments.pix_to_face[N, y, x, k]) != -1]
                        )))

    # get the corresponding visaible vertices and clean up
    vertex_list = []
    for i in face_list:
        vertices_for_element = faces[i]
        vertex_list.extend(vertices_for_element)
    vertex_list = list(sorted(set(vertex_list)))

    # Render the scene of needed
    if render:
        texture_base = torch.ones_like(mesh.verts_packed().cpu()) # N X 3
        texture = texture_base.cpu() * torch.tensor([0.5, 0.5, 1]).cpu()
        texture[vertex_list] = torch.tensor([1.0, 0, 0])
        texture = pytorch3d.renderer.TexturesVertex(texture.unsqueeze(0))
        mesh.textures = texture
        fig = plot_scene({
            "Figure 1":{
                "Mesh": mesh,
                "Camera": camera
            }
        })
        fig.show()

    return vertex_list

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

    # # Question 5.1
    # data = load_rgbd_data()
    # # Split the data into respective images
    # data1 = {}
    # data2 = {}
    # for key in data:
    #     if key[-1]  == '1':
    #         data1[key[:-1]] = data[key]
    #     else:
    #         data2[key[:-1]] = data[key]
    
    # point1, color1 = unproject_depth_image(torch.tensor(data1['rgb']), torch.tensor(data1['mask']), torch.tensor(data1['depth']), data1['cameras'])
    # point2, color2 = unproject_depth_image(torch.tensor(data2['rgb']), torch.tensor(data2['mask']), torch.tensor(data2['depth']), data2['cameras'])

    # renderpointcloud_360(points = point1, colors = color1, filepath = "results/pointcloud1.gif", flip = True)
    # renderpointcloud_360(points = point2, colors = color2, filepath = "results/pointcloud2.gif", flip = True)

    # pointcloud1 = pytorch3d.structures.Pointclouds(points=point1.unsqueeze(0), features=color1.unsqueeze(0))
    # pointcloud2 = pytorch3d.structures.Pointclouds(points=point2.unsqueeze(0), features=color2.unsqueeze(0))
    # pointcloud3 = pytorch3d.structures.join_pointclouds_as_scene([pointcloud1, pointcloud2])

    # renderpointcloud_360(points = pointcloud3.points_list()[0], 
    #                      colors = pointcloud3.features_list()[0],
    #                        filepath = "results/pointcloud3.gif", flip = True)

    # # Question 5.2
    # points, color = generateTorus(num_samples=200)
    # renderpointcloud_360(points = points, colors = color, filepath = "results/torus_parametric.gif")

    # # Question 5.3
    # mesh = generateTorusMesh()
    # render360('results/torus_implicit.gif', camera_dist=7, model=(mesh.verts_padded().squeeze(0), mesh.faces_padded().squeeze(0)))

    # Question 6
    # The idea here is to create a function that can return a visility matrix for a given point cloud from a given camera position
    # This function has direct relation to my research work and that is why I have chosen to implement it here
    image_size = 512

    renderer = get_mesh_renderer(image_size=512)
    img_list = []

    vertices, faces = utils.load_cow_mesh(path="data/cow.obj")
    mesh = pytorch3d.structures.Meshes(verts=vertices.unsqueeze(0), faces=faces.unsqueeze(0))
    increment = 10
    for i in tqdm(range(0, 360+1, increment)):
        # For Camera 1
        R, T = pytorch3d.renderer.look_at_view_transform(dist = 2, elev = 0, azim = i)
        camera1 = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T)

        # For Camera 2
        R, T = pytorch3d.renderer.look_at_view_transform(dist = 5, elev = 30, azim = i - increment)
        camera2 = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T) 

        visMat = findVisibilityMatrix(mesh = mesh, camera = camera1, render = False)
        texture_base = torch.ones_like(mesh.verts_packed().cpu()) # N X 3
        texture = texture_base.cpu() * torch.tensor([0.5, 0.5, 1]).cpu()
        texture[visMat] = torch.tensor([1.0, 0, 0])
        texture = pytorch3d.renderer.TexturesVertex(texture.unsqueeze(0))
        mesh.textures = texture

        image = renderer(mesh, cameras=camera2)
        image = image.cpu().numpy()[0, ..., :3]
        img_list.append(Image.fromarray((image * 255).astype(np.uint8)))

    imageio.mimsave("results/visibility.gif", img_list, fps=5)












