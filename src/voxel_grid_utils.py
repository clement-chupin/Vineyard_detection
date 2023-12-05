import torch
import math
import torch.nn as nn
from multiprocessing.pool import ThreadPool as Pool



import time
import os
import sys


#if you want to import file insides others folders like "import include.toto_utils"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_advanced_utils import LocalMaxSearching
from math_advanced_utils import Dens_compute


from ros_topic_utils import point_cloud_3d
from ros_topic_utils import point_cloud_4d

class VoxelGrid:
	def __init__(self,
				pointcloud=torch.rand(10,3),
				polar_grid=True,
				nb_chunk = 16,
				nb_process = 16,
				angular_remover = -0.6,
				height_absolute_treshold = 0.05,
				height_relative_treshold = 0.0,
				dist_voxel = 0.5,
				angular_voxel = 20,
				dist_voxel_pad = 0.0,
				angular_voxel_pad = 0.0,
				size_voxels = 2.0,
				size_voxels_padding = 0.0,
				size_pi_pcc = 0.2,
				density_angular_poi = -100.0
				 ):
		
		self.polar_grid = polar_grid
		self.use_gpu = True
		if torch.cuda.is_available() and self.use_gpu:  
			dev = "cuda:0" 
		else:  
			dev = "cpu"  
		self.device = torch.device(dev)
		self.dens_compute_model = Dens_compute(0.2)
		self.nb_chunk = nb_chunk
		self.nb_process = nb_process

		self.zeros_tens_1 = torch.zeros(0,1,device=self.device)
		self.zeros_tens_3 = torch.zeros(0,3,device=self.device)
		self.zeros_tens_4 = torch.zeros(0,4,device=self.device)

		self.angular_remover = angular_remover
		self.height_absolute_treshold = height_absolute_treshold
		self.height_relative_treshold = height_relative_treshold


		self.dist_voxel        = dist_voxel
		self.angular_voxel     = angular_voxel
		self.dist_voxel_pad    = dist_voxel_pad
		self.angular_voxel_pad = angular_voxel_pad

		self.size_voxels = size_voxels
		self.size_voxels_padding = size_voxels_padding

		self.size_pi_pcc = size_pi_pcc

		self.density_angular_poi = density_angular_poi
		self.density_to_pt_interest = LocalMaxSearching(self.density_angular_poi,0.0)
		self.ground_segmentator = LocalMaxSearching(-self.angular_remover,self.height_absolute_treshold,)

		self.pointcloud = pointcloud
		if self.polar_grid:
			self.index_tiles = self.pointcloud_to_tiles_polar(pointcloud)
		else:
			self.index_tiles = self.pointcloud_to_tiles_grid(pointcloud)

	def pointcloud_to_tiles_grid(self,pointcloud):
		min_x,min_y = pointcloud[:,:2].min(0).values[:]
		max_x,max_y = pointcloud[:,:2].max(0).values[:]
		#print("size x : ",max_x-min_x)
		#print("size y : ",max_y-min_y)
		tile_indices = torch.zeros((pointcloud.shape[0], 6), dtype=int, device=self.device)

		tile_indices[:, 0] = ((pointcloud[:, 0] - min_x) / self.size_voxels).floor()
		tile_indices[:, 1] = ((pointcloud[:, 1] - min_y) / self.size_voxels).floor()

		tile_indices[:, 2] = ((pointcloud[:, 0] - min_x - self.size_voxels_padding) / self.size_voxels).floor()
		tile_indices[:, 3] = ((pointcloud[:, 1] - min_y - self.size_voxels_padding) / self.size_voxels).floor()

		tile_indices[:, 4] = ((pointcloud[:, 0] - min_x + self.size_voxels_padding) / self.size_voxels).floor()
		tile_indices[:, 5] = ((pointcloud[:, 1] - min_y + self.size_voxels_padding) / self.size_voxels).floor()

		return tile_indices
	
	def pointcloud_to_tiles_polar(self,pointcloud):
		distance = torch.sqrt((pointcloud**2).sum(1))
		min_dist = distance.min(0).values
		angle = torch.atan2(pointcloud[:,0],pointcloud[:,1])
		min_pi = angle.min(0).values
		tile_indices = torch.zeros((pointcloud.shape[0], 6), dtype=int, device=self.device)

		tile_indices[:, 0] =  ((distance - min_dist) / math.sqrt(self.dist_voxel)      ).floor()
		tile_indices[:, 1] =  ((angle    - min_pi)   / math.radians(self.angular_voxel)).floor()%(360/self.angular_voxel)

		nb_x,nb_y = torch.max(tile_indices[:,0])+1,torch.max(tile_indices[:,1])+1

		tile_indices[:, 2] =  (((distance - min_dist - math.sqrt(self.dist_voxel_pad))       / math.sqrt(self.dist_voxel)      ).floor()+nb_x)%(nb_x)
		tile_indices[:, 3] =  (((angle    - min_pi   - math.radians(self.angular_voxel_pad)) / math.radians(self.angular_voxel)).floor()+nb_y)%(nb_y)

		tile_indices[:, 4] =  (((distance - min_dist + math.sqrt(self.dist_voxel_pad))       / math.sqrt(self.dist_voxel)      ).floor()+nb_x)%(nb_x)
		tile_indices[:, 5] =  (((angle    - min_pi   + math.radians(self.angular_voxel_pad)) / math.radians(self.angular_voxel)).floor()+nb_y)%(nb_y)
		
		return tile_indices	
	def get_poincloud_by_tile_index(self,index_tiles,i_tile,pointcloud):
		master_index = (index_tiles[:,:2] == i_tile)
		master_index = master_index[:,0]*master_index[:,1]
		master_tile_pc = pointcloud[master_index]

		padding_index_a = (index_tiles[:,2:4] == i_tile)
		padding_index_b = (index_tiles[:,4:6] == i_tile)
		padding_index   = (padding_index_a[:,0]+padding_index_b[:,0])*(padding_index_a[:,1]+padding_index_b[:,1])
		padding_tile_pc = pointcloud[padding_index]

		return master_tile_pc,padding_tile_pc
	
	def gr_main(self,pointcloud,pointcloud_padding):
		return self.ground_segmentator(
			pointcloud,
			pointcloud_padding,
			-pointcloud[:,2],
			-pointcloud_padding[:,2],
			)

	def density_main(self,pointcloud,pointcloud_padding):
		result = self.dens_compute_model(pointcloud,pointcloud_padding)
		# print(result.shape)
		return result
	
	def pt_interest_main(self,pointcloud,pointcloud_padding):
		# color = self.gr_main(pointcloud,pointcloud_padding)
		# non_ground_with_padding = pointcloud[color[:,0]==0.0]
		# if len(non_ground_with_padding) == 0:
		# 	return self.zeros_tens_4
		# index_tiles = self.pointcloud_to_tiles_polar(pointcloud)
		density_pointcloud = self.density_main(pointcloud,pointcloud_padding )
		
		pt_interest = self.density_to_pt_interest(
			pointcloud,
			pointcloud,
			density_pointcloud.view(-1),
			density_pointcloud.view(-1),)
		# print(pt_interest.shape)
		return pt_interest

	def ditribute_operation_over_voxels(self,operation):
		nb_x,nb_y = torch.max(self.index_tiles[:,0])+1,torch.max(self.index_tiles[:,1])+1
		self.pointcloud.share_memory_()
		def generic_call(i_tile,):
			master_tile_pc,padding_tile_pc = self.get_poincloud_by_tile_index(self.index_tiles,i_tile,self.pointcloud)
			if len(master_tile_pc) == 0:
				return self.zeros_tens_4	
			result =  operation(master_tile_pc,padding_tile_pc)
			return torch.hstack([master_tile_pc,result])
		
		xs,ys = torch.meshgrid(torch.arange(0,nb_x,1,device=self.device),torch.arange(0,nb_y,1,device=self.device))
		list_index = torch.stack([xs.view(nb_x,nb_y),ys.view(nb_x,nb_y)],2).view(nb_x*nb_y,2)
		with Pool(self.nb_process) as p:
			computed_pointcloud = p.map(generic_call, list_index,self.nb_chunk)
			computed_pointcloud = torch.vstack(computed_pointcloud)
		return computed_pointcloud
	
	def get_density(self,):
		return self.ditribute_operation_over_voxels(self.density_main)
	def get_pt_interest(self,):
		return self.ditribute_operation_over_voxels(self.pt_interest_main)
	def get_ground_segmentation(self,):
		return self.ditribute_operation_over_voxels(self.gr_main)



# t3 = time.time()
# if self.debug:
# 	print("- a in : ",(t1-t0)*1000,"ms")
# 	print("- b in : ",(t2-t1)*1000,"ms")
# 	print("- c in : ",(t3-t2)*1000,"ms")
