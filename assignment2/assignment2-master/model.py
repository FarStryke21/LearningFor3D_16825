from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d


class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)
    
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        self.extra_feature = torch.meshgrid(torch.arange(-1.,1.,2./32),torch.arange(-1.,1.,2./32),torch.arange(-1.,1.,2./32))
        # print(self.extra_feature.shape)
        self.extra_feature = torch.cat([self.extra_feature[0].unsqueeze(0),self.extra_feature[1].unsqueeze(0),self.extra_feature[2].unsqueeze(0)],axis=0).to(self.device)
        self.extra_feature = self.extra_feature.unsqueeze(0)
        
        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            # TODO:
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024), # b x 512 -> # b x 2048
                nn.Linear(1024, 8*6**3),
                nn.Unflatten(-1, (8, 6, 6, 6)),
                nn.ConvTranspose3d(8,8,3,stride=1, padding=0, output_padding=0, groups=1, bias=True,dilation=1),
                nn.BatchNorm3d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(True),
                nn.ConvTranspose3d(8, 16, 3, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1),
                nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(True),
                nn.ConvTranspose3d(16, 32, 3, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1),
                nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(True),
                nn.ConvTranspose3d(32, 16, 5, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1),
                nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(True),
                nn.ConvTranspose3d(16, 8, 5, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1),
                nn.BatchNorm3d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(True),
                nn.ConvTranspose3d(8, 4, 7, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1),
                nn.BatchNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(True),
                nn.ConvTranspose3d(4, 4, 7, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1),
                nn.BatchNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(True),
                nn.ConvTranspose3d(4, 1, 1, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
            )
            # self.decoder =  
            # self.decoder = nn.Sequential(
            #     # Attempt 1
            #     nn.Linear(512, 512*2*2*2), # b x 512 -> # b x 2048
            #     nn.Unflatten(-1, (512, 2, 2, 2)), # b x 2048 -> b x 256 x 2 x 2 x 2
            #     nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, bias=False, padding=1), # b x 256 x 2 x 2 x 2 -> b x 128 x 2 x 2 x 2
            #     nn.BatchNorm3d(256),
            #     nn.ReLU(),
            #     nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1), # b x 128 x 2 x 2 x 2 -> b x 64 x 2 x 2 x 2
            #     nn.BatchNorm3d(128),
            #     nn.ReLU(),
            #     nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1), # b x 64 x 2 x 2 x 2 -> b x 32 x 2 x 2 x 2
            #     nn.BatchNorm3d(64),
            #     nn.ReLU(),
            #     nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1), # b x 32 x 2 x 2 x 2 -> b x 8 x 2 x 2 x 2
            #     nn.BatchNorm3d(32),
            #     # nn.ReLU(),
            #     # nn.ConvTranspose3d(32, 16, kernel_size=1, bias=False), # b x 8 x 2 x 2 x 2 -> b x 1 x 2 x 2 x 2
            #     # nn.BatchNorm3d(16),
            #     # nn.ReLU(),
            #     # nn.ConvTranspose3d(16, 8, kernel_size=1, bias=False), # b x 8 x 2 x 2 x 2 -> b x 1 x 2 x 2 x 2
            #     # nn.BatchNorm3d(8),
            #     # nn.ReLU(),
            #     nn.ConvTranspose3d(32, 1, kernel_size=1, bias=False), # b x 8 x 2 x 2 x 2 -> b x 1 x 2 x 2 x 2
            #     # nn.Sigmoid()
            # )   
           
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            # self.decoder = 
            self.decoder = nn.Sequential(
                nn.Linear(512, self.n_point),
                nn.LeakyReLU(),
                nn.Linear(self.n_point, self.n_point*3),
                ReshapeLayer((-1, args.n_points, 3)),
                nn.Tanh()
            )
                            
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            # self.decoder =
            self.decoder = nn.Sequential(
                nn.Linear(512, 3*mesh_pred.verts_packed().shape[0]),
                nn.Tanh()
            )

        elif args.type == 'implicit':
            # Input : b x (512 + 3)
            # Output : b x 1

            # self.decoder = nn.Sequential(
            #     nn.Linear(512 + 3, 128),
            #     nn.Linear(128, 64),
            #     nn.Linear(64, 1),
            # )
            # self.decoder = nn.Sequential(
            # nn.Conv3d(in_channels=512 + 3, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            # nn.Sigmoid()  # Applying Sigmoid to ensure output is between 0 and 1
            # )
            # self.decoder = nn.Sequential(
            #     nn.Linear(515, 512),  # 512 (image features) + 3 (3D coordinates)
            #     nn.ReLU(),
            #     nn.Linear(512, 256),
            #     nn.ReLU(),
            #     nn.Linear(256, 1),  # Output: occupancy value
            #     nn.Sigmoid()
            # )
            self.decoder = nn.Sequential(
                nn.Linear(512 + 3, 2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 4096),
                nn.LeakyReLU(),
                nn.Linear(4096, 1),
                nn.Sigmoid()
            )
            


        elif args.type == 'parametric':
            # Input : b x (n_points x 2)
            # Output : b x (n_points x 3)
            self.n_points = args.n_points
            self.decoder = nn.Sequential(
                nn.Linear(self.n_points * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.n_points * 3),
                nn.Tanh()
            )

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            voxels_pred = self.decoder(encoded_feat)  
            # print(f'voxel pred : {voxels_pred.shape}')            
            return voxels_pred

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat)   # 2, 5000, 3            
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)                          
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

        elif args.type == "parametric":
            # for each image in a batch, sample n_points from the image, creating a tensor of size (B, n_points, 2)
            image_width = images.shape[2]
            image_height = images.shape[1]
            random_points = torch.rand(B, 1000, 2)  
            random_points[:, :, 0] *= image_width  # Scale x-coordinate to image width
            random_points[:, :, 1] *= image_height  # Scale y-coordinate to image height

            # Convert to integer coordinates
            # random_points = random_points.type(torch.int)

            # Ensure points are within image boundaries
            random_points[:, :, 0] = torch.clamp(random_points[:, :, 0], 0, image_width - 1)
            random_points[:, :, 1] = torch.clamp(random_points[:, :, 1], 0, image_height - 1)

            
            # convert to a tensor of (B, n_points * 2)
            pointclouds_pred = random_points.view(B, -1).to(args.device).double()
            print(pointclouds_pred.dtype)

            # pass through the decoder
            parametric_pred = self.decoder(pointclouds_pred)

            # reshape to (B, n_points, 3)
            parametric_pred = parametric_pred.view(B, self.n_points, 3)


            return pointclouds_pred



        elif args.type == "implicit":
            # sample 1000 points from the voxel grid pf 32x32x32 nomralized to -1 to 1

            coords = torch.linspace(-1, 1, 32)
            meshgrid = torch.stack(torch.meshgrid(coords, coords, coords), -1)  # Size: (32, 32, 32, 3)
            meshgrid = meshgrid.reshape(-1, 3)  # Size: (32768, 3)

            # Add sampled_points to encoded_feat
            encoded_feat_expanded = encoded_feat.unsqueeze(1).expand(-1, meshgrid.size(1), -1)  # Size: (B, 32768, 512)
            inputs = torch.cat([encoded_feat_expanded, meshgrid], dim=-1)  # Size: (B, 32768, 512 + 3)
            input = inputs.view(-1, 515)
            # Pass through the decoder
            occupancy = self.decoder(inputs)  # Size: (B*32768, 1)

            occupancy = occupancy.view(B, meshgrid.size(1), -1)  # Size: (B, 32768, 1)
            # Reshape occupancy to match the desired output shape
            occupancy = occupancy.permute(0, 2, 1).view(B, 32, 32, 32)  # Size: (B, 1, 32, 32, 32)

            return occupancy

            # voxels_in = self.decoder_in(encoded_feat).view(-1,1,24,24,24)
            # voxels_pred = self.decoder(voxels_in)
            # self.extra_feature_ = self.extra_feature.repeat(voxels_in.shape[0],1,1,1,1)
            # voxels_pred = torch.cat((voxels_pred,self.extra_feature_),axis=1)
            # voxels_pred = self.decoder_out(voxels_pred)
            # # print("Here")
            # return voxels_pred
            # image_features = encoded_feat.unsqueeze(1)  # Size becomes (b, 1, 512)

            # # Expand dimensions to match the coordinates size
            # image_features = image_features.expand(-1, 32768, -1).to(self.device)  # Size becomes (b, 32768, 512)

            # coords = torch.linspace(-1, 1, 32)
            # meshgrid = torch.stack(torch.meshgrid(coords, coords, coords), -1)  # Size: (32, 32, 32, 3)
            # meshgrid = meshgrid.reshape(-1, 3)  # Size: (32768, 3)

            # # Repeat the meshgrid for each item in the batch
            # meshgrid = meshgrid.unsqueeze(0).repeat(encoded_feat.size(0), 1, 1).to(self.device)  # Size: (b, 32768, 3)
            # x = torch.cat([image_features, meshgrid], dim=-1)  # Concatenate image features and 3D coordinates
            # x = x.view(-1, 515)
            # occupancy = self.decoder(x)  # Pass through the
            # occupancy = occupancy.view(-1, 1, 32, 32, 32)
            # return occupancy
        
            # # print("Shape of B: "+str(B))
            # grid_size = 32
            # x = torch.linspace(-1, 1, grid_size, dtype=torch.float32)
            # y = torch.linspace(-1, 1, grid_size, dtype=torch.float32)
            # z = torch.linspace(-1, 1, grid_size, dtype=torch.float32)
            # meshgrid = torch.meshgrid(x, y, z)
            # # print(f"Meshgrid initial: {meshgrid.shape}")
            # meshgrid = torch.stack(meshgrid, dim=-1).reshape(-1, 3).to(args.device)  # Reshape to (32*32*32, 3)
            # # print(f"Meshgrid Update: {meshgrid.shape}")

            # --------------------------------------------------------
            # Tile image_feature to match the shape of meshgrid
            # Tile encoded_feat to match the batch size
            # image_feature_tiled = encoded_feat.unsqueeze(2).repeat(1, 1, meshgrid.size(0))

            # # Reshape meshgrid to have the same number of columns as the tiled encoded features
            # meshgrid_reshaped = meshgrid.unsqueeze(0).repeat(B, 1, 1)

            # # Concatenate image_feature_tiled and meshgrid
            # inputs = torch.cat([image_feature_tiled, meshgrid_reshaped.permute(0, 2, 1)], dim=1)
            # # print(f"Input Shape: {inputs.shape}")
            # # input shape is B x (512 + 3) x 32*32*32
            # # Decoder takes input of 
            # inputs = inputs.view(B, 515, 32, 32, 32)
            # # Reshape output to match the batch size and meshgrid size
            # implicit_pred = self.decoder(inputs)

            # return implicit_pred

            # --------------------------------------------------------

            # Process encoded features
            # encoded_feat = self.fc_encoded(encoded_feat)
            # encoded_feat = nn.ReLU()(encoded_feat)

            # # Process coordinates
            # self.n_coords = args.n_coords
            # self.sample_p = 2 * torch.rand((B, self.num_coords * 3)) - 1
            # coordinates = self.fc_coords(self.sample_p.to(args.device))
            # coordinates = nn.ReLU()(coordinates)

            # # Concatenate encoded features and coordinates
            # combined = torch.cat((encoded_feat, coordinates), dim=1)

            # # Pass through combined fully connected layers
            # occupancy = self.fc_combined(combined)

            # return occupancy

            # --------------------------------------------------------

            # self.n_coords = args.n_coords # default=2048
            # self.fc_p = nn.Linear(3*self.n_coords, 128)
            # self.fc_c = nn.Linear(512, 128)

            # self.sample_p = 2 * torch.rand((B, self.num_coords * 3)) - 1

            # net = self.fc_p(self.sample_p)
            # net_c = self.fc_c(encoded_feat).unsqueeze(1)
            # net = net + net_c

            # implicit_pred = self.decoder(net).squeeze(-1)

            # return implicit_pred
