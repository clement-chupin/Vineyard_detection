import torch
import torch.nn as nn
from math_advanced_utils import Dens_compute
from math_advanced_utils import LocalMaxSearching
class OctreeTree:
	def __init__(self,
			  pointcloud,
			  kwargs
			  ):
		
		self.octo_npoints_max = 1000
		self.octo_depth_max = 100
		self.octo_padding_size = 0.05

		self.ground_segmentation_a = -0.6
		self.ground_segmentation_b = 0.05

		self.density_kernel_size = 0.1
		self.density_max_a = -100.0
		self.density_max_b =  0.0

		kwargs_keys = kwargs.keys()
		if "octo_npoints_max" in kwargs_keys:
			self.octo_npoints_max = kwargs["octo_npoints_max"]
		if "octo_depth_max" in kwargs_keys:
			self.octo_depth_max = kwargs["octo_depth_max"]
		if "octo_padding_size" in kwargs_keys:
			self.octo_padding_size = kwargs["octo_padding_size"]


		if "ground_segmentation_a" in kwargs_keys:
			self.ground_segmentation_a = kwargs["ground_segmentation_a"]
		if "ground_segmentation_b" in kwargs_keys:
			self.ground_segmentation_b = kwargs["ground_segmentation_b"]
		if "density_max_a" in kwargs_keys:
			self.density_max_a = kwargs["density_max_a"]
		if "density_max_b" in kwargs_keys:
			self.density_max_b = kwargs["density_max_b"]
		if "density_kernel_size" in kwargs_keys:
			self.density_kernel_size = kwargs["density_kernel_size"]
			

		self.device = pointcloud.device
		self.size_poincloud = (pointcloud.max(0).values - pointcloud.min(0).values)#.max()

		self.binary_octree_coding = ((torch.arange(8,device=self.device).view(-1,1).repeat(1,3)/torch.tensor([4,2,1],device=self.device))%2).int().bool()
		self.raw_octree_coding = (1-self.binary_octree_coding*2).int()

		self.dens_compute = Dens_compute(self.density_kernel_size)
		self.max_dens_compute = LocalMaxSearching(self.density_max_a,self.density_max_b)
		self.ground_segmentation_compute = LocalMaxSearching(self.ground_segmentation_a,self.ground_segmentation_b)
		
		self.idx_i = 0
		self.master_node = OctreeNode(self,1,self.size_poincloud,pointcloud.mean(0),pointcloud,pointcloud,self.octo_padding_size)
	
	def compute(self,operation_str):
		return self.master_node.distribute_operation_over_leaf(operation_str)


class OctreeNode:
	def __init__(self,tree,depth,size,position,pointcloud,pointcloud_padded,padding_size):
		# print("--")
		# print("pc received : ",pointcloud.shape)
		# print("pc paddeddd : ",pointcloud_padded.shape)
		self.leaf_node = []
		self.tree = tree
		self.depth = depth
		self.size = size
		self.device = pointcloud.device
		self.voxel_size = self.size/2
		self.position = position
		self.log_acti = False
		self.pading_size = padding_size
		self.dens_compute = tree.dens_compute
		self.max_dens_compute = tree.max_dens_compute
		self.ground_segmentation_compute = tree.ground_segmentation_compute

		if len(pointcloud) > self.tree.octo_npoints_max and self.depth < tree.octo_depth_max:
			new_positions = (position+(self.tree.raw_octree_coding*self.voxel_size/2)).to(torch.float32)
			select_xyz = pointcloud <  position
			selection_pad_a = pointcloud_padded <  (position + self.pading_size)
			selection_pad_b = pointcloud_padded <  (position - self.pading_size);del position
			for voxel_position,voxel_code in zip(new_positions,self.tree.binary_octree_coding):
				select = (select_xyz == voxel_code.view(1,-1)).prod(-1).bool()
				select_pad = (selection_pad_a == voxel_code.view(1,-1)) + (selection_pad_b == voxel_code.view(1,-1))
				select_pad = select_pad.prod(-1).bool()
				if select.sum(-1) != 0:
					oc_node = OctreeNode(
						self.tree,
						depth+1,
						self.size/2,voxel_position,
						pointcloud[select],
						pointcloud_padded[select_pad],
						padding_size,)
					self.leaf_node.append(oc_node)
			del new_positions,select_xyz
		else:
			self.pointcloud = pointcloud
			self.pointcloud_padded = pointcloud_padded

	def distribute_operation_over_leaf(self,operation):
		if len(self.leaf_node) == 0:
			return getattr(self, operation)()
		else:
			result_all_leaf = []
			for leaf in self.leaf_node:
				result_leaf = leaf.distribute_operation_over_leaf(operation)
				result_all_leaf.append(result_leaf)
			return torch.vstack(result_all_leaf)
		
	def get_entier_pc(self):
		return self.pointcloud
	def get_colored_pc(self):
		self.tree.idx_i+=1
		return torch.hstack([
			self.pointcloud,
			torch.ones((self.pointcloud.shape[0],1),device=self.pointcloud.device)*self.tree.idx_i%9])

	def get_density(self,):
		self.pointcloud_density = self.dens_compute(self.pointcloud,self.pointcloud_padded)
		return torch.hstack([self.pointcloud,self.pointcloud_density])
	
	def get_local_max(self,):
		self.pointcloud_max_density = self.max_dens_compute(self.pointcloud,self.pointcloud,self.pointcloud_density,self.pointcloud_density,)
		return torch.hstack([self.pointcloud,self.pointcloud_max_density])
	
	def get_ground_segmentation(self,):
		self.pointcloud_max_density = self.ground_segmentation_compute(self.pointcloud,self.pointcloud_padded,-self.pointcloud[:,2],-self.pointcloud_padded[:,2])
		return torch.hstack([self.pointcloud,self.pointcloud_max_density])
