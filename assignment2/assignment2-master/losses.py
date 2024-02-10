import torch
import pytorch3d as p3d

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# Convert src and tgt to a 2d shape of b and number of voxels
	voxel_src_new = voxel_src.view(voxel_src.size(0),-1)
	voxel_tgt_new = voxel_tgt.view(voxel_tgt.size(0),-1)

	# Implement loss
	loss_obj = torch.nn.BCEWithLogitsLoss()
	# implement some loss for binary voxel grids
	loss = loss_obj(voxel_src_new, voxel_tgt_new)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3 
	# Compute pairwise distances
	dist_src_tgt, src_tgt_idx, _ = p3d.ops.knn.knn_points(point_cloud_src, point_cloud_tgt, K=1) # (batch_size, n_points, 1)
	dist_tgt_src, tgt_src_idx, _ = p3d.ops.knn.knn_points(point_cloud_tgt, point_cloud_src, K=1) # (batch_size, n_points, 1)
	
	
	loss_chamfer = torch.sum(torch.stack([dist_src_tgt, dist_tgt_src]))

	# implement chamfer loss from scratch
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	# Get the vertex positions of the source mesh
	V = mesh_src.verts_packed()

	# Compute the Laplacian operator on the vertex positions
	L = mesh_src.laplacian_packed()

	# Compute the smoothness loss as the mean of the squared Laplacian operator
	loss_laplacian = torch.square(torch.linalg.norm(torch.matmul(L, V)))
	return loss_laplacian
