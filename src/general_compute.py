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
from ros_topic_utils import ros_topic_manager
# mp.set_sharing_strategy('file_system')
# mp.set_start_method('fork')
from multiprocessing.pool import ThreadPool as Pool
# set_start_method('spawn')
from ros_topic_utils import born_lidar_range
from ros_topic_utils import bord_lidar_height

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ros_topic_utils import point_cloud_3d
from ros_topic_utils import point_cloud_4d
from voxel_grid_utils import VoxelGrid
import time




class Density_compute:
	def __init__(self):

		self.activate                 = rospy.get_param('~activate', True)
		self.use_gpu                  = rospy.get_param('~use_gpu', True)
		self.debug               = rospy.get_param('~debug', True)

		self.mode_detection           = rospy.get_param('~computation_target', "gr") #["density,"ground_segmentation","pt_interest"]

		self.size_pcc                 = rospy.get_param('~density_pcc', 0.3)
		self.size_poi_pcc              = rospy.get_param('~density_poi_pcc', 0.3)
		self.size_poi_max_angular              = rospy.get_param('~density_poi_max_agular',-100)
		self.frame_id_link              		  = rospy.get_param('~link',"velo_link")
		

		
		#self.dens_compute_model = torch.jit.script(self.dens_compute_model)
		#self.dens_compute_model = torch.compile(self.dens_compute_model)

		
		self.angular_voxel_pad        = rospy.get_param('~angle_voxel_split_padding', 5)
		self.dist_voxel_pad           = rospy.get_param('~dist_voxel_split_padding', 0.1)
		self.angular_voxel            = rospy.get_param('~angle_voxel_split', 20)
		self.dist_voxel               = rospy.get_param('~dist_voxel_split', 0.5)

		self.size_voxels         	  = rospy.get_param('~size_voxels', 0.1)
		self.size_voxels_padding      = rospy.get_param('~size_voxels_padding', 0)


		self.nb_process               = rospy.get_param('~pool_n_workers', 128)
		self.nb_chunk                 = rospy.get_param('~pool_n_chunk', 32)

		self.range_lidar_min          = rospy.get_param('~lidar_range_min', 0.0)
		self.range_lidar_max          = rospy.get_param('~lidar_range_max', 100.0)
		self.z_lidar_min_limit		  = rospy.get_param('~lidar_z_min', -2.0)

		self.height_relative_treshold = rospy.get_param('~gr_relative_treshold', 1.0)
		self.height_absolute_treshold = rospy.get_param('~gr_absolute_treshold', 0.01)
		self.angular_remover          = rospy.get_param('~gr_angular',0.6)

		self.polar_grid               = rospy.get_param('~polar_grid',True)
		self.ros_topic_manager = ros_topic_manager()

		### Try to speed up pytorch computation
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


			

	def compute_pointcloud(self,pointcloud):
		voxel_grid_pointcloud=  VoxelGrid(
			pointcloud,
			self.polar_grid,
			self.nb_chunk,
			self.nb_process,
			self.angular_remover,
			self.height_absolute_treshold,
			self.height_relative_treshold,
			self.dist_voxel,
			self.angular_voxel,
			self.dist_voxel_pad,
			self.angular_voxel_pad,
			self.size_voxels,
			self.size_voxels_padding,
			self.size_poi_pcc,
			self.size_poi_max_angular
			)

		if self.mode_detection == "density":
			density_pointcloud =  voxel_grid_pointcloud.get_density() 
			print("density : ",density_pointcloud.shape)
			self.ros_topic_manager.pub_message("pointcloud_density",point_cloud_4d(density_pointcloud.cpu(), self.frame_id_link))

		if self.mode_detection == "pt_interest":
			pt_interest_pointcloud =  voxel_grid_pointcloud.get_pt_interest() 
			print("pt_interest : ",pt_interest_pointcloud.shape)
			self.ros_topic_manager.pub_message("pointcloud_pt_interest",point_cloud_3d(pt_interest_pointcloud.cpu(), self.frame_id_link))

		if self.mode_detection == "ground_segmentation":
			ground_segmentation_pointcloud =  voxel_grid_pointcloud.get_ground_segmentation()
			print("ground_segmentation : ",ground_segmentation_pointcloud.shape)
			self.ros_topic_manager.pub_message("pointcloud_ground_segmentation",point_cloud_4d(ground_segmentation_pointcloud.cpu(), self.frame_id_link))



	def map_callback(self,msg: PointCloud2):
			t0 = time.time()
			pointcloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
			if len(pointcloud):
				pointcloud = torch.tensor(pointcloud,dtype=torch.float32,device=self.device)
				self.ros_topic_manager.pub_message("raw_pointcloud_received",point_cloud_3d(pointcloud.cpu(), self.frame_id_link))
				self.compute_pointcloud(pointcloud)
				t1 = time.time()
				if self.debug:
					print(pointcloud.shape[0],"points")
					print(round(1/(t1-t0)),"Hz")
					print(round((t1-t0)*1000),"ms")


if __name__ == '__main__':
	rospy.init_node('density_compute')
	
	a = Density_compute()
	rospy.spin()




