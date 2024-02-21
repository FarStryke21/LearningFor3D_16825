import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh', 'implicit', 'parametric'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

# Write a function that computes the saliency map of the input image given the model path
def compute_saliency_map(model_path, image_gt, output_path, args):
    # Load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SingleViewto3D()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Compute the saliency map
    saliency = torch.zeros(image_gt.shape[2], image_gt.shape[3])
    for i in range(image_gt.shape[2]):
        for j in range(image_gt.shape[3]):
            image_copy = image_gt.clone()
            image_copy[0, :, i, j] = 0
            output = model(image_copy, args)
            saliency[i, j] = torch.norm(output - model(image_gt, args))

    # Save the saliency map
    plt.imsave(output_path, saliency.cpu().detach().numpy(), cmap='hot')

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    
    images_gt, mesh_gt = preprocess(feed_dict, args)
    
    for type in ['vox', 'point', 'mesh']:
        model_path = f"checkpoint_{type}.pth"
        output_path = f'data/saliency_{type}.png'
        compute_saliency_map(model_path, images_gt, output_path, args)

    compute_saliency_map(args)