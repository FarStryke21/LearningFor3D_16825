import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def knn(x, k):

    inner = - 2* torch.bmm(x.transpose(1,2), x)

    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    return idx


def get_graph_feature(x, k, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2) #TODO: change?
    num_dims = x.size(1)
    x = x.reshape(batch_size, -1, num_points).contiguous()

    idx = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device='cuda').view(-1, 1, 1)*num_points

    idx = idx + idx_base
    idx = idx.flatten()
    x = x.transpose(1,2).contiguous()
    feature = x.reshape(batch_size*num_points, -1)[idx, :].contiguous()
    feature = feature.reshape(batch_size, num_points, k, num_dims).contiguous()  # B x N x K x D
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1,1,k,1).contiguous()
    feature = torch.cat((feature-x, x), dim=3)
    feature = feature.permute(0,3,1,2)
  
    return feature

class cls_model(nn.Module):
    def __init__(self, num_classes=3, k=3):
        super(cls_model, self).__init__()
        self.k = k
        self.dropout = 0.3
        self.embed_dim = 1024

        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(6,64, kernel_size=1)

        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64*2,64, 1, bias=False)

        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(64*2,128, 1, bias=False)

        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(128*2,256, 1, bias=False)

        self.bn5 = nn.BatchNorm2d(self.embed_dim)
        self.conv5 = nn.Conv1d(512,self.embed_dim, 1, bias=False)

        self.relu = nn.LeakyReLU()

        self.linear1 = nn.Linear(self.embed_dim, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0,2,1)

        x = get_graph_feature(x, k=self.k)
        x = self.relu(self.bn1(self.conv1(x)))
        x1 = torch.max(x, dim=-1, keepdim=False)[0]


        x = get_graph_feature(x1, k=self.k)
        x = self.relu(self.bn2(self.conv2(x)))
        x2 = torch.max(x, dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.relu(self.bn3(self.conv3(x)))
        x3 = torch.max(x, dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.relu(self.bn4(self.conv4(x)))
        x4 = torch.max(x, dim=-1, keepdim=False)[0]

        x = torch.cat((x1,x2,x3,x4), dim=1)

        x = self.conv5(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        x = self.linear1(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.dp1(x)

        x = self.linear2(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.dp2(x)

        x = self.linear3(x)
        return x