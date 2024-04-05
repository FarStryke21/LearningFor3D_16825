import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg

from tqdm import tqdm

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    # ------ TO DO: Make Prediction ------
    batch_size = 16
    num_batch = (test_data.shape[0] // batch_size)+1
    pred_label = torch.zeros_like(test_label)

    for i in tqdm(range(num_batch)):
        pred = model(test_data[i*batch_size: (i+1)*batch_size].to(args.device))
        curr_pred_label = torch.argmax(pred, -1, keepdim=False).cpu()
        pred_label[i*batch_size: (i+1)*batch_size, :] = curr_pred_label

    pred_label = torch.Tensor(pred_label).cpu()
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    print ("Visualizing Segmentation Results")
    for idx in tqdm(range(len(test_data))):
    # Visualize Segmentation Result (Pred VS Ground Truth)
        viz_seg(test_data[args.i], test_label[args.i], "{}/seg/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
        viz_seg(test_data[args.i], pred_label[args.i], "{}/seg/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)
