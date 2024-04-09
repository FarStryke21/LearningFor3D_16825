import numpy as np
import argparse

import torch
from models import cls_model
import models_dgcnn
from utils import create_dir, viz_seg

from tqdm import tqdm

import pytorch3d
from data_loader import get_data_loader

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='cls', help='The task to perform (cls or seg)')
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    parser.add_argument('--use_dgcnn', action='store_true', help='Use DGCNN for classification task', default=False)
    parser.add_argument('--batch_size', type=int, default=32)

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    if args.use_dgcnn:
        model = models_dgcnn.cls_model().to(args.device)
    else:
        model = cls_model().to(args.device)
    
    # Load Model Checkpoint
    if args.use_dgcnn:
        model_path = './checkpoints_dgcnn/cls/{}.pt'.format(args.load_checkpoint)
    else:
        model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    # test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    # test_label = torch.from_numpy(np.load(args.test_label))

    # --------------Rotate the data----------------
    test_dataloader = get_data_loader(args=args, train=False)
    rot = torch.tensor([20,0,0])
    R = pytorch3d.transforms.euler_angles_to_matrix(rot, 'XYZ')
    test_dataloader.dataset.data = (R @ test_dataloader.dataset.data.transpose(1, 2)).transpose(1, 2)

    test_data = test_dataloader.dataset.data
    test_label = test_dataloader.dataset.label

    # ------ TO DO: Make Prediction ------
    batch_size = args.batch_size
    num_batch = (test_data.shape[0] // batch_size)+1
    pred_label = []

    for i in tqdm(range(num_batch)):
        pred = model(test_data[i*batch_size: (i+1)*batch_size].to(args.device))
        curr_pred_label = torch.argmax(pred, -1, keepdim=False).cpu()
        curr_pred_label = list(curr_pred_label)
        pred_label.extend(curr_pred_label)

    # Compute Accuracy
    pred_label = torch.Tensor(pred_label).cpu()
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    print ("Visualizing the results")
    for idx in tqdm(range(len(test_data))):
        viz_seg(test_data[idx].cpu(), test_label[idx].cpu(), "{}/cls/gt_{}_{}.gif".format(args.output_dir, args.exp_name, idx), args.device, points=args.num_points)
        viz_seg(test_data[idx].cpu(), pred_label[idx].cpu(), "{}/cls/pred_{}_{}.gif".format(args.output_dir, args.exp_name, idx), args.device, points=args.num_points)
    
    # get a list of the predicted labels which were incorrect
    test_label = test_label.cpu().numpy()
    pred_label = pred_label.cpu().numpy()

    incorrect_labels = []
    for i in range(len(test_label)):
        if test_label[i] != pred_label[i]:
            incorrect_labels.append(i)
    
    print("Incorrect labels: ", incorrect_labels)

# Baseline Output:
# test accuracy: 0.9790136411332634
# Incorrect labels:  [406, 618, 650, 651, 664, 670, 673, 685, 707, 708, 714, 716, 721, 726, 827, 832, 859, 864, 883, 916]

# 5000 points experiment
# test accuracy: 0.9769150052465897
# Incorrect labels:  [406, 618, 650, 651, 663, 664, 667, 670, 673, 685, 707, 708, 714, 716, 721, 726, 827, 832, 859, 864, 883, 916]

# 1000 point experiment
# test accuracy: 0.9716684155299056
# Incorrect labels:  [406, 618, 644, 651, 663, 664, 667, 670, 671, 673, 685, 707, 708, 714, 716, 721, 726, 787, 803, 806, 827, 832, 859, 864, 870, 883, 916]

# DGCNN Output:
# test accuracy: 0.9674711437565582
# Incorrect labels:  [445, 518, 618, 619, 620, 642, 650, 651, 660, 664, 670, 676, 680, 685, 703, 706, 707, 716, 772, 781, 787, 796, 827, 839, 858, 859, 864, 869, 883, 922, 944]


