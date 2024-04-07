import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):

    inner = - 2* torch.bmm(x.transpose(1,2), x)

    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    return idx


def knn_graph(x, k, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2) #TODO: change?
    num_dims = x.size(1)
    x = x.reshape(batch_size, -1, num_points).contiguous()

    idx = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device='cuda').view(-1, 1, 1)*num_points

    idx = idx + idx_base
    idx = idx.flatten()
    x = x.transpose(1,2).contiguous()
    # TODO: check idx size and see if any reshaping is needed
    # TODO: check x size and see if any reshaping is needed
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.reshape(batch_size*num_points, -1)[idx, :].contiguous()
    feature = feature.reshape(batch_size, num_points, k, num_dims).contiguous()  # B x N x K x D
    # TODO: convert x = B x N x 1 x D to shape x = B x N x k x D (hint: repeating the elements in that dimension)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1,1,k,1).contiguous()
    feature = torch.cat((feature-x, x), dim=3)
    feature = feature.permute(0,3,1,2)
  
    return feature

# def knn_graph(x, k=10):
#     B, D, N = x.shape
#     print(f"x.shape: {x.shape}")
#     dists = torch.cdist(x, x)
#     print(f"dists.shape: {dists.shape}")
#     _, inds = torch.topk(dists, k=k+1, dim=1, largest=False)
#     inds = inds[:, 1:]
#     print(f"inds.shape: {inds.shape}")
#     inds += torch.arange(0, x.shape[0], device="cuda").view(-1, 1, 1)*x.shape[-1]
#     inds = inds.reshape(-1)
#     print(f"inds.shape: {inds.shape}")
#     x = x.transpose(2, 1).contiguous()
#     print(f"x.shape: {x.shape}")
#     feats = x.reshape(B*N, -1)[:, inds]
#     print(f"feats.shape: {feats.shape}")
#     feats = feats.reshape(B, N, k, D) 
#     print(f"feats.shape: {feats.shape}")
#     x = x.unsqueeze(2).repeat(1, 1, k, 1)
#     print(f"x.shape: {x.shape}")

#     feats = torch.cat((feats-x, x), dim=-1).permute(0, 3, 1, 2).contiguous()
#     print(f"feats.shape: {feats.shape}")

#     return feats

class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()

        self.conv1 = torch.nn.Conv2d(6, 64, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, bias=False)
        self.conv3 = torch.nn.Conv2d(64, 128, 1, bias=False)
        self.conv4 = torch.nn.Conv2d(128, 256, 1, bias=False)
        self.conv5 = torch.nn.Conv2d(512, 1024, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.bn5 = torch.nn.BatchNorm2d(1024)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        out1 = knn_graph(points)
        out1 = torch.amax(F.LeakyReLU(self.bn1(self.conv1(out1)), negative_slope=0.2), dim=-1, keepdim=False)

        out2 = knn_graph(out1)
        out2 = torch.amax(F.LeakyReLU(self.bn2(self.conv2(out2)), negative_slope=0.2), dim=-1, keepdim=False)

        out3 = knn_graph(out2)
        out3 = torch.amax(F.LeakyReLU(self.bn3(self.conv3(out3)), negative_slope=0.2), dim=-1, keepdim=False)

        out4 = knn_graph(out3)
        out4 = torch.amax(F.LeakyReLU(self.bn4(self.conv4(out4)), negative_slope=0.2), dim=-1, keepdim=False)

        out = torch.cat((out1, out2, out3, out4), dim=1)
        
        out = torch.amax(F.LeakyReLU(self.bn5(self.conv5(out)), negative_slope=0.2), dim=-1, keepdim=False)

        out = self.fc(out)

        return out


class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()

        self.conv1 = nn.Conv2d(6, 64, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 1, bias=False)
        self.conv3 = nn.Conv2d(64*2, 64, 1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, 1, bias=False)
        self.conv5 = nn.Conv2d(64*2, 64, 1, bias=False)
        self.conv6 = nn.Conv1d(192, 1024, 1, bias=False)
        self.conv7 = nn.Conv1d(16, 64, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(64)

        self.point_layer = nn.Sequential(
            nn.Conv1d(1280, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Conv1d(256, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, num_seg_classes, 1, bias=False),
        )


    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        out1 = knn_graph(points)
        out1 = F.LeakyReLU(self.bn1(self.conv1(out1)), negative_slope=0.2)
        out1 = torch.amax(F.LeakyReLU(self.bn2(self.conv2(out1)), negative_slope=0.2), dim=-1, keepdim=False)

        out2 = knn_graph(out1)
        out1 = F.LeakyReLU(self.bn3(self.conv3(out1)), negative_slope=0.2)
        out2 = torch.amax(F.LeakyReLU(self.bn4(self.conv4(out2)), negative_slope=0.2), dim=-1, keepdim=False)

        num_points = points.size(1)
        out3 = knn_graph(out2)
        out3 = torch.amax(F.LeakyReLU(self.bn5(self.conv5(out3)), negative_slope=0.2), dim=-1, keepdim=False)

        out_comb1 = torch.cat((out1, out2, out3), dim=1)

        out4 = knn_graph(out_comb1)
        out4 = torch.amax(F.LeakyReLU(self.bn6(self.conv6(out4)), negative_slope=0.2), dim=-1, keepdim=False)

        cat_vet = F.LeakyReLU(self.bn7(self.conv7(cat_vet.view(points.shape[0], -1, 1))), negative_slope=0.2)

        out_comb2 = torch.cat((out4, cat_vet), dim=1).repeat(1, 1, num_points) 

        out = torch.cat((out1, out2, out3, out_comb2), dim=1)

        out = self.fc(out)

        return out
