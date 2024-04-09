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
    rot = torch.tensor([30,0,0])
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

# Rotate 10
# test accuracy: 0.4616998950682057
# Incorrect labels:  [0, 1, 2, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 27, 28, 30, 31, 32, 34, 35, 36, 37, 38, 39, 41, 44, 45, 46, 49, 51, 52, 55, 57, 58, 59, 62, 63, 66, 67, 68, 69, 71, 73, 75, 76, 77, 83, 84, 86, 87, 88, 89, 91, 92, 94, 95, 97, 98, 99, 101, 102, 103, 104, 107, 108, 110, 112, 113, 114, 115, 118, 119, 120, 121, 124, 126, 128, 129, 131, 133, 134, 135, 136, 137, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151, 152, 153, 154, 156, 159, 160, 161, 162, 165, 166, 167, 168, 169, 170, 172, 173, 174, 176, 177, 178, 179, 181, 183, 184, 185, 188, 189, 190, 191, 192, 193, 194, 196, 197, 199, 201, 202, 205, 207, 208, 209, 210, 211, 213, 215, 216, 217, 218, 220, 221, 223, 224, 225, 226, 227, 228, 230, 231, 233, 234, 236, 238, 239, 240, 241, 242, 243, 244, 245, 247, 250, 251, 253, 255, 258, 261, 262, 264, 266, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 284, 285, 286, 287, 288, 289, 290, 291, 293, 294, 295, 296, 298, 299, 300, 301, 302, 304, 305, 307, 308, 309, 311, 312, 313, 314, 315, 317, 318, 319, 320, 321, 324, 325, 326, 327, 329, 331, 333, 334, 337, 338, 339, 340, 343, 344, 345, 347, 348, 350, 351, 352, 353, 354, 355, 356, 359, 360, 362, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 377, 380, 381, 382, 383, 385, 387, 391, 393, 396, 398, 400, 401, 404, 405, 406, 408, 410, 411, 412, 413, 414, 415, 416, 419, 421, 422, 423, 424, 425, 427, 428, 431, 433, 434, 435, 438, 439, 440, 442, 444, 445, 446, 447, 449, 450, 451, 452, 453, 455, 456, 457, 459, 461, 462, 463, 464, 465, 466, 469, 470, 471, 472, 473, 474, 477, 478, 479, 480, 481, 484, 485, 486, 488, 490, 491, 495, 496, 498, 499, 500, 501, 503, 507, 511, 512, 516, 517, 518, 519, 520, 521, 522, 524, 525, 527, 530, 531, 532, 533, 534, 535, 536, 537, 539, 540, 542, 543, 544, 550, 551, 552, 555, 556, 557, 559, 560, 561, 563, 564, 565, 566, 568, 569, 570, 572, 573, 574, 576, 579, 580, 581, 582, 584, 585, 586, 588, 589, 591, 592, 593, 594, 597, 598, 599, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 616, 618, 619, 620, 621, 622, 623, 624, 627, 629, 631, 632, 635, 637, 639, 642, 644, 645, 646, 647, 648, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 663, 665, 667, 668, 669, 670, 671, 672, 673, 675, 676, 677, 678, 680, 681, 682, 684, 685, 686, 687, 688, 690, 691, 693, 695, 697, 698, 701, 702, 703, 704, 706, 708, 709, 712, 714, 715, 716, 718, 746, 750, 759, 770, 777, 816, 824, 827, 830, 858, 864, 883, 891, 915, 916, 944]

# Rotate 30
# test accuracy: 0.3462749213011542
# Incorrect labels:  [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 25, 26, 27, 28, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 51, 52, 53, 54, 57, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 101, 102, 103, 104, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 119, 120, 121, 122, 124, 128, 129, 131, 132, 133, 134, 135, 136, 137, 139, 140, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 153, 154, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 181, 182, 183, 184, 185, 188, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 205, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 220, 221, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 236, 237, 238, 239, 240, 241, 242, 243, 245, 246, 247, 248, 250, 251, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 270, 271, 272, 274, 275, 276, 277, 278, 279, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 296, 297, 298, 299, 300, 301, 302, 304, 305, 307, 308, 309, 310, 311, 312, 313, 314, 315, 317, 318, 319, 321, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 338, 339, 340, 341, 343, 344, 345, 346, 347, 348, 350, 352, 353, 354, 355, 356, 357, 358, 359, 360, 362, 364, 365, 366, 367, 368, 369, 370, 371, 372, 374, 375, 376, 377, 379, 380, 381, 382, 383, 385, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 398, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 414, 415, 416, 419, 420, 421, 422, 424, 425, 426, 427, 428, 430, 431, 432, 433, 434, 435, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 464, 465, 466, 467, 469, 470, 471, 472, 473, 474, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 488, 490, 491, 492, 493, 494, 495, 497, 498, 499, 500, 501, 502, 503, 504, 505, 507, 509, 511, 512, 515, 516, 517, 518, 519, 521, 522, 523, 524, 525, 526, 527, 530, 531, 532, 533, 534, 535, 536, 537, 539, 540, 542, 543, 544, 548, 549, 550, 552, 553, 554, 555, 556, 557, 558, 559, 561, 562, 563, 565, 566, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 591, 592, 593, 594, 595, 596, 597, 598, 599, 601, 602, 603, 604, 606, 607, 608, 609, 610, 612, 614, 616, 619, 623, 627, 632, 634, 639, 646, 647, 648, 651, 652, 653, 656, 658, 659, 662, 663, 664, 665, 670, 673, 676, 685, 686, 687, 688, 690, 693, 695, 697, 699, 703, 706, 708, 715, 716, 719, 727, 731, 732, 734, 736, 738, 742, 744, 750, 756, 758, 759, 762, 763, 765, 767, 770, 773, 779, 781, 786, 787, 789, 790, 794, 799, 800, 803, 808, 809, 813, 814, 818, 819, 820, 822, 826, 827, 828, 832, 834, 836, 839, 849, 853, 855, 856, 858, 862, 864, 865, 867, 870, 872, 877, 878, 879, 880, 883, 890, 891, 898, 901, 903, 904, 907, 912, 914, 916, 917, 922, 925, 927, 931, 936, 943, 945, 946, 947, 950]

# DGCNN Output:
# test accuracy: 0.9674711437565582
# Incorrect labels:  [445, 518, 618, 619, 620, 642, 650, 651, 660, 664, 670, 676, 680, 685, 703, 706, 707, 716, 772, 781, 787, 796, 827, 839, 858, 859, 864, 869, 883, 922, 944]


