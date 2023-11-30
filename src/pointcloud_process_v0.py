#!/usr/bin/env python3
import numpy as np
import torch
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from rospy.numpy_msg import numpy_msg
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import sys
import os
from torch.multiprocessing import Queue,Pool,Process, set_start_method
import torch.multiprocessing as mp
import math
import ros_numpy
import torch.nn as nn

# mp.set_sharing_strategy('file_system')
# mp.set_start_method('fork')
from multiprocessing.pool import ThreadPool as Pool
# set_start_method('spawn')


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ros_topic_utils import point_cloud_3d
from ros_topic_utils import point_cloud_4d

import time

class Dens_compute(nn.Module):
	def __init__(self, dist):
		super().__init__()
		self.d = dist
	def forward(self,pointcloud,pointcloud_padding):
		return (torch.cdist(pointcloud_padding,pointcloud)<=self.d).sum(0).view(-1,1)[:pointcloud.shape[0]]



class Density_compute:
	def __init__(self):

		self.activate                 = rospy.get_param('~activate', True)
		self.use_gpu                  = rospy.get_param('~use_gpu', True)
		self.full_debug               = rospy.get_param('~debug', True)

		self.mode_detection           = rospy.get_param('~computation_target', "gr") #["density,"gr","points_of_interest"]

		self.size_pcc                 = rospy.get_param('~density_pcc', 0.3)
		self.size_pi_pcc              = rospy.get_param('~density_pi_pcc', 0.3)
		self.link              		  = rospy.get_param('~link',"velo_link")
		

		self.dens_compute_model = Dens_compute(self.size_pcc)
		#self.dens_compute_model = torch.jit.script(self.dens_compute_model)
		#self.dens_compute_model = torch.compile(self.dens_compute_model)

		
		self.angular_voxel_pad        = rospy.get_param('~angle_voxel_split_padding', 5)
		self.dist_voxel_pad           = rospy.get_param('~dist_voxel_split_padding', 0.1)
		self.angular_voxel            = rospy.get_param('~angle_voxel_split', 20)
		self.dist_voxel               = rospy.get_param('~dist_voxel_split', 0.5)


		self.nb_process               = rospy.get_param('~pool_n_workers', 128)
		self.nb_chunk                 = rospy.get_param('~pool_n_chunk', 32)

		self.range_lidar_min          = rospy.get_param('~lidar_range_min', 0.0)
		self.range_lidar_max          = rospy.get_param('~lidar_range_max', 100.0)
		self.z_lidar_min_limit		  = rospy.get_param('~lidar_z_min', -2.0)

		self.height_relative_treshold = rospy.get_param('~gr_relative_treshold', 1.0)
		self.height_absolute_treshold = rospy.get_param('~gr_absolute_treshold', 0.01)
		self.angular_remover          = rospy.get_param('~gr_angular',0.6)

		
		self.size_voxels         = 0.2
		self.size_voxels_padding = 0.1


		mp.set_start_method('forkserver')

		torch.set_grad_enabled(False)


		torch.jit.enable_onednn_fusion(True)
		torch.backends.cudnn.benchmark = True
		torch._C._jit_set_autocast_mode(False)


		if self.activate:
			if torch.cuda.is_available() and self.use_gpu:  
				dev = "cuda:0" 
			else:  
				dev = "cpu"  
			self.device = torch.device(dev)

			self.pointcloud_sub = rospy.Subscriber('~pointcloud', PointCloud2, self.map_callback,queue_size=1)
			self.pc_cutted_pub  = rospy.Publisher('~pointcloud_voxel_split', PointCloud2,queue_size=1)
			self.pc_density_pub = rospy.Publisher('~pointcloud_density', PointCloud2,queue_size=1)
			self.pc_gr_pub      = rospy.Publisher('~pointcloud_ground_removal', PointCloud2,queue_size=1)
			self.pc_pts_interest_pub = rospy.Publisher('~pointcloud_pt_interest', PointCloud2,queue_size=1)
			self.pc_pts_interest_only_pub = rospy.Publisher('~pointcloud_pt_interest_only', PointCloud2,queue_size=1)

			self.zeros_tens_1 = torch.zeros(0,1,device=self.device)
			self.zeros_tens_3 = torch.zeros(0,3,device=self.device)
			self.zeros_tens_4 = torch.zeros(0,4,device=self.device)



	def local_maxima_main(self,pointcloud,pointcloud_padding,z_i=None,z_j=None,a=1.0,tresh_abs=0.0,tresh_rel=0.0):
		P_i,P_j = pointcloud,pointcloud_padding #P_i etudiés, par rapport au projection des P_j
		distance_ij = torch.cdist(P_i[:,:2],P_j[:,:2])  
		if z_i is None or z_j is None:
			z_i = P_i[:,2]
			z_j = P_j[:,2]
		P_j_proj = distance_ij*a+z_j

		P_j_max  = P_j_proj.max(-1).values
		
		dist_i_to_max = P_j_max-z_i

		seg_color = (dist_i_to_max <= (tresh_abs+tresh_rel*dist_i_to_max.mean())).view(-1,1)
		return seg_color*1.0
	

	def gr_main(self,pointcloud,pointcloud_padding):
		return self.local_maxima_main(
			-pointcloud,
			-pointcloud_padding,
			None,
			None,
			-self.angular_remover,
			self.height_absolute_treshold,
			self.height_relative_treshold
			)
	

	
	def density_main(self,pointcloud,pointcloud_padding,pcc_dist=None):
		return self.dens_compute_model(pointcloud,pointcloud_padding)
		if pcc_dist is None:
			pcc_dist = self.size_pcc
		
		result = (torch.cdist(pointcloud_padding,pointcloud)<=pcc_dist).sum(0).view(-1,1)[:pointcloud.shape[0]]

		return result

	def pt_interest_main(self,pointcloud,pointcloud_padding):

		color = self.gr_main(pointcloud_padding,pointcloud_padding)
		non_ground_with_padding = pointcloud_padding[color[:,0]==0.0]
		if len(non_ground_with_padding) == 0:
			return self.zeros_tens_3,self.zeros_tens_1
		
		# index_tiles = self.pointcloud_to_tiles_polar(pointcloud)
		density_non_ground = self.density_main(non_ground_with_padding,pointcloud_padding,self.size_pi_pcc )
		#return non_ground_with_padding,density_non_ground


		pt_interest = self.local_maxima_main(
			non_ground_with_padding,
			non_ground_with_padding,
			density_non_ground.view(-1),
			density_non_ground.view(-1),
			-100.0,0.0,0.0
			)
		return non_ground_with_padding,pt_interest

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
	
	#@torch.compile()
	def pointcloud_to_tiles_polar(self,pointcloud):
		def dist_mod(x):
			return x
			return torch.sqrt(x)
		distance = dist_mod(torch.sqrt((pointcloud**2).sum(1)))
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
	def born_lidar_range(self,pointcloud):
		selection_a = torch.sqrt((pointcloud**2).sum(1)) >= self.range_lidar_min
		selection_b = torch.sqrt((pointcloud**2).sum(1)) <= self.range_lidar_max
		return pointcloud[selection_a*selection_b]
	
	def bord_lidar_height(self,pointcloud):
		return pointcloud[pointcloud[:,2] > self.z_lidar_min_limit]

	
	#@torch.compile()
	def split_and_compute(self,pointcloud,index_tiles):
		nb_x,nb_y = torch.max(index_tiles[:,0])+1,torch.max(index_tiles[:,1])+1
		pointcloud.share_memory_()
		t0 = time.time()
		xs,ys = torch.meshgrid(torch.arange(0,nb_x,1,device=self.device),torch.arange(0,nb_y,1,device=self.device))
		list_index = torch.stack([xs.view(nb_x,nb_y),ys.view(nb_x,nb_y)],2).view(nb_x*nb_y,2)
		t1 = time.time()

		def split_and_calculate_density_pointcloud(i_tile,):
			return generic_call(i_tile,self.density_main)
		def split_and_calculate_gr_pointcloud(i_tile,):
			return generic_call(i_tile,self.gr_main)
		def split_and_calculate_pt_interest_pointcloud(i_tile,):
			return generic_call(i_tile,self.pt_interest_main,True)
		
		def generic_call(i_tile,func,pointcloud_returned=False):
			master_tile_pc,padding_tile_pc = self.get_poincloud_by_tile_index(index_tiles,i_tile,pointcloud)
			if len(master_tile_pc) == 0:
				return self.zeros_tens_4
			if pointcloud_returned:
				pc,result =  func(master_tile_pc,padding_tile_pc)
				return torch.hstack([pc,result])
			else:
				result =  func(master_tile_pc,padding_tile_pc)
				return torch.hstack([master_tile_pc,result])
			
		
		if self.mode_detection == "density":
			with Pool(self.nb_process) as p:
				pointcloud_computed = p.map(split_and_calculate_density_pointcloud, list_index,self.nb_chunk)
			pointcloud_computed = torch.vstack(pointcloud_computed)
			t2 = time.time()
			self.pc_density_pub.publish(point_cloud_4d(pointcloud_computed.cpu(), self.link))

		if self.mode_detection == "gr":
			with Pool(self.nb_process) as p:
				pointcloud_computed = p.map(split_and_calculate_gr_pointcloud, list_index,self.nb_chunk)
			pointcloud_computed = torch.vstack(pointcloud_computed)
			t2 = time.time()
			self.pc_gr_pub.publish(point_cloud_4d(pointcloud_computed.cpu(), self.link))

		if self.mode_detection == "pt_interest":
			with Pool(self.nb_process) as p:
				pointcloud_computed = p.map(split_and_calculate_pt_interest_pointcloud, list_index,self.nb_chunk)
			pointcloud_computed = torch.vstack(pointcloud_computed)
			t2 = time.time()
			self.pc_pts_interest_pub.publish(point_cloud_4d(pointcloud_computed.cpu(), self.link))

			pt_interest_only = pointcloud_computed[pointcloud_computed[:,3]==1.0]
			self.pc_pts_interest_only_pub.publish(point_cloud_4d(pt_interest_only.cpu(), self.link))
		

		t3 = time.time()
		if self.full_debug:
			print("- a in : ",(t1-t0)*1000,"ms")
			print("- b in : ",(t2-t1)*1000,"ms")
			print("- c in : ",(t3-t2)*1000,"ms")

		torch.cuda.empty_cache()


	def map_callback(self,msg: PointCloud2):

		t0 = time.time()
		pointcloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
		


		print(pointcloud.shape)
		if len(pointcloud):
		
			pointcloud = torch.tensor(pointcloud,dtype=torch.float32,device=self.device)
			# self.pc_cutted_pub.publish(point_cloud_3d(pointcloud.detach().cpu(), self.link))


			

			#pointcloud = torch.stack([pointcloud[:,0],pointcloud[:,2],pointcloud[:,1],],1)
			# t01 = time.time()
			# RT = torch.tensor([
			# 	[1.0,0.0,0.0],
			# 	[0.0,0.0,-0.8],
			# 	[0.0,1.0,0.0]
			# ],device=self.device)
			# pointcloud = pointcloud @ RT

			


			pointcloud = self.born_lidar_range(pointcloud)
			pointcloud = self.bord_lidar_height(pointcloud)

			# pointcloud+= torch.rand_like(pointcloud)*0.1
			# print(pointcloud.shape)



			# t02 = time.time()
			index_tiles = self.pointcloud_to_tiles_polar(pointcloud)
			# #index_tiles = self.pointcloud_to_tiles_grid(pointcloud)

			seg_pc = torch.hstack([pointcloud,(index_tiles[:,0]%2+index_tiles[:,1]%5).view(-1,1)]).cpu()
			self.pc_cutted_pub.publish(point_cloud_4d(seg_pc, self.link))
			self.split_and_compute(pointcloud,index_tiles)

			t1 = time.time()
			if self.full_debug or True:
				print("Total computed in : ",(t1-t0)*1000,"ms")

		
	
if __name__ == '__main__':
	rospy.init_node('density_compute')
	
	a = Density_compute()
	rospy.spin()




