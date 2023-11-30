#!/usr/bin/env python3
import numpy as np
import torch
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from rospy.numpy_msg import numpy_msg
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
import sys
import os
# from torch.multiprocessing import Queue,Pool,Process, set_start_method
# import torch.multiprocessing as mp
# import math
# import random
import ros_numpy
import torch.nn as nn
import time

# from multiprocessing.pool import ThreadPool as Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ros_topic_utils import point_cloud_3d
from ros_topic_utils import point_cloud_4d

from octotree_utils import OctreeTree
from octotree_utils import LocalMaxSearching
from octotree_utils import Dens_compute

from ros_topic_utils import ros_topic_manager
from ros_topic_utils import publish_lines


from math_advanced_utils import grid_based_on_param
from math_advanced_utils import plan_approx
from math_advanced_utils import error_of_approx
from math_advanced_utils import ground_approx
from math_advanced_utils import grid_based_on_param_poly
from math_advanced_utils import plan_approx_poly
from math_advanced_utils import error_of_approx_poly
from math_advanced_utils import ground_approx_poly
from math_advanced_utils import proj_pc_on_plan
from math_advanced_utils import merge_by_distance
from math_advanced_utils import foot_selection



class ProcessPointcloud:
	def __init__(self):

		self.activate                 = rospy.get_param('~activate', True)
		self.use_gpu                  = rospy.get_param('~use_gpu', True)
		self.full_debug               = rospy.get_param('~debug', True)

		self.mode_detection           = rospy.get_param('~computation_target', "gr") #["density,"gr","points_of_interest"]

		# self.size_pcc                 = rospy.get_param('~density_pcc', 0.3)
		# self.size_pi_pcc              = rospy.get_param('~density_pi_pcc', 0.3)
		self.frame_id_link              		  = rospy.get_param('~link',"velo_link")
		

		self.range_lidar_min          = rospy.get_param('~lidar_range_min', 0.0)
		self.range_lidar_max          = rospy.get_param('~lidar_range_max', 100.0)
		self.z_lidar_min_limit		  = rospy.get_param('~lidar_z_min', -2.0)

		self.octo_npoints_max         = rospy.get_param('~octo_npoints_max', 1000)
		self.octo_depth_max           = rospy.get_param('~octo_depth_max', 100)
		self.octo_padding_size        = rospy.get_param('~octo_padding_size', 0.05)

		self.reduce_pointcloud_ratio  = rospy.get_param('~reduce_pointcloud_ratio', 0.5)

		self.transfo_link = rospy.get_param('~transfo_link', [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
		
		self.ground_segmentation_a    = rospy.get_param('~ground_segmentation_a', -0.6)
		self.ground_segmentation_b    = rospy.get_param('~ground_segmentation_b', 0.05)
		
		self.density_max_a            = rospy.get_param('~density_max_a', -100)
		self.density_max_b            = rospy.get_param('~density_max_b', 0.0)
		self.density_kernel_size      = rospy.get_param('~density_kernel_size', 0.1)

		self.vine_foot_to_plane_trhld = rospy.get_param('~vine_foot_to_plane_trhld', 0.2)
		self.final_vine_selection_a   = rospy.get_param('~final_vine_selection_a', -0.5)
		self.final_vine_selection_b   = rospy.get_param('~final_vine_selection_b', 0.0)


		self.ground_approx_error      = rospy.get_param('~ground_approx_error', 0.2)
		self.final_merging_distance   = rospy.get_param('~final_merging_distance', 0.1)


		self.foot_selection_pcc_kernel   = rospy.get_param('~foot_selection_pcc_kernel', 0.1)
		self.foot_selection_heigth_threshold = rospy.get_param('~foot_selection_heigth_threshold', 0.2)


		self.pointcloud_octree_kwargs = {
			"octo_npoints_max" : self.octo_npoints_max,
			"octo_depth_max" : self.octo_depth_max,
			"octo_padding_size" : self.octo_padding_size,
			
			"ground_segmentation_a" : self.ground_segmentation_a,
			"ground_segmentation_b" : self.ground_segmentation_b,
			
			"density_max_a" : self.density_max_a,
			"density_max_b" : self.density_max_b,
			"density_kernel_size" : self.density_kernel_size,
		}
		self.plants_octree_kwargs = self.pointcloud_octree_kwargs


		self.plants_poi_octree_kwargs = {
			"octo_npoints_max" : self.octo_npoints_max,
			"octo_depth_max" : self.octo_depth_max,
			"octo_padding_size" : self.octo_padding_size,
			
			"ground_segmentation_a" : self.ground_segmentation_a,
			"ground_segmentation_b" : self.ground_segmentation_b,
			
			"density_max_a" : -100,
			"density_max_b" : self.density_max_b,
			"density_kernel_size" : 0.3,
		}



		self.ros_topic_manager = ros_topic_manager()
		
		# mp.set_start_method('forkserver')

		torch.set_grad_enabled(False)
		self.final_point_selection = LocalMaxSearching(self.final_vine_selection_a,self.final_vine_selection_b)
		# self.dense_compute = Dens_compute()
		
		
		if self.activate:
			if torch.cuda.is_available() and self.use_gpu:  
				dev = "cuda:0" 
			else:  
				dev = "cpu"  
			self.device = torch.device(dev)

			self.transfo_link_3d = torch.tensor(self.transfo_link,device=self.device)
			self.transfo_link_4d = torch.hstack([torch.vstack(
				[
					self.transfo_link_3d,
					torch.tensor([[0,0,0]],device=self.device)]),torch.tensor([0,0,0,1.0],device=self.device).view(-1,1)])

			self.pointcloud_sub = rospy.Subscriber('~pointcloud', PointCloud2, self.map_callback,queue_size=1)
			
			self.zeros_tens_1 = torch.zeros(0,1,device=self.device)
			self.zeros_tens_3 = torch.zeros(0,3,device=self.device)
			self.zeros_tens_4 = torch.zeros(0,4,device=self.device)




	
	def born_lidar_range(self,pointcloud):
		selection_a = torch.sqrt((pointcloud**2).sum(1)) >= self.range_lidar_min
		selection_b = torch.sqrt((pointcloud**2).sum(1)) <= self.range_lidar_max
		return pointcloud[selection_a*selection_b]
	
	def bord_lidar_height(self,pointcloud):
		return pointcloud[pointcloud[:,2] > self.z_lidar_min_limit]
	

	def hard_reduce_pointcloud(self,pointcloud,divide_ratio):
		return pointcloud[torch.rand(pointcloud.shape[0],device=pointcloud.device)<divide_ratio]
	


	def split_and_compute(self,pointcloud):
		t0 = time.time()
		self.ros_topic_manager.pub_message("pointcloud_transformed" ,point_cloud_3d(torch.matmul(pointcloud,self.transfo_link_3d).cpu(), "base_link"))
		
		pointcloud = self.hard_reduce_pointcloud(pointcloud,self.reduce_pointcloud_ratio)
		# print("hard reduce of : ",self.reduce_pointcloud_ratio)

		pointcloud = self.born_lidar_range(pointcloud)
		t10 = time.time()
		octree = OctreeTree(torch.matmul(pointcloud,self.transfo_link_3d),self.pointcloud_octree_kwargs)
		t20 = time.time()
		colored_vox  = octree.compute("get_colored_pc")
		t30 = time.time()
		ground_plant = octree.compute("get_ground_segmentation")
		t40 = time.time()
		pointcloud_octree = octree.compute("get_entier_pc")
		t50 = time.time()
	
		ground_pointcloud = pointcloud_octree[ground_plant[:,3]==1]#[:,:2]
		plants_pointcloud = pointcloud_octree[ground_plant[:,3]==0]#[:,:2]

		ground_params = ground_approx_poly(ground_pointcloud,self.ground_approx_error)
		t60 = time.time()
		plan_approximed = grid_based_on_param_poly(ground_params,pointcloud.device)
		t61 = time.time()
		octree_plant = OctreeTree(plants_pointcloud,self.plants_octree_kwargs)
		t70 = time.time()
		plants_octree   = octree_plant.compute("get_entier_pc")
		t80 = time.time()
		density_plant   = octree_plant.compute("get_density")
		t90 = time.time()
		local_max_plant = octree_plant.compute("get_local_max")
		t100 = time.time()
		point_of_interest = plants_octree[local_max_plant[:,3]==1]



		octree_poi_plant = OctreeTree(point_of_interest,self.plants_poi_octree_kwargs)

		# octree_pointcloud = octree_poi_plant.compute("get_entier_pc")
		density_poi_plant = octree_poi_plant.compute("get_density")
		plants_detected  = octree_poi_plant.compute("get_local_max")
		plants_detected = plants_detected[plants_detected[:,3]==1]

		# print(plants_detected[:,3])




		t110 = time.time()
		if len(point_of_interest):
			t120 = time.time()
			is_foot = self.final_point_selection(point_of_interest,point_of_interest,-point_of_interest[:,2],-point_of_interest[:,2])[:,0] == 1
			t130 = time.time()
			detected_points = point_of_interest[is_foot]

			error_of_proj = torch.abs(error_of_approx(detected_points,ground_params))
			t140 = time.time()
			detected_points = detected_points[error_of_proj < self.vine_foot_to_plane_trhld]
			t150 = time.time()

			detected_points = merge_by_distance(detected_points,self.final_merging_distance)
			final_detected_points = detected_points[foot_selection(detected_points,point_of_interest,self.foot_selection_pcc_kernel,self.foot_selection_heigth_threshold,self.frame_id_link)]
			t160 = time.time()
			self.ros_topic_manager.pub_message("detected_points"  ,point_cloud_3d(torch.matmul(detected_points,self.transfo_link_3d.T).cpu(), self.frame_id_link))
			self.ros_topic_manager.pub_message("final_detected_points"  ,point_cloud_3d(torch.matmul(final_detected_points,self.transfo_link_3d.T).cpu(), self.frame_id_link))
			t170 = time.time()

			# print("t10 to t00 : ",(t10-t0)*1000,"ms")
			# print("t20 to t10 : ",(t20-t10)*1000,"ms")
			# print("t30 to t20 : ",(t30-t20)*1000,"ms")
			# print("t40 to t30 : ",(t40-t30)*1000,"ms")
			# print("t50 to t40 : ",(t50-t40)*1000,"ms")
			# print("t60 to t50 : ",(t60-t50)*1000,"ms")
			# print("61 to t60 : ",(t61-t60)*1000,"ms")
			# print("t70 to t61 : ",(t70-t61)*1000,"ms")
			# print("t80 to t670 : ",(t80-t70)*1000,"ms")
			# print("t90 to t0   : ",(t90-t80)*1000,"ms")
			# print("t100 to t0 : ",(t100-t90)*1000,"ms")
			# print("t110 to t0 : ",(t110-t100)*1000,"ms")
			# print("t120 to t0 : ",(t120-t110)*1000,"ms")
			# print("t130 to t0 : ",(t130-t120)*1000,"ms")
			# print("t140 to t0 : ",(t140-t130)*1000,"ms")
			# print("t150 to t0 : ",(t150-t140)*1000,"ms")
			# print("t160 to t0 : ",(t160-t150)*1000,"ms")
			# print("t170 to t0 : ",(t170-t160)*1000,"ms")

			# print("total process : ",(t170-t0)*1000,"ms")



		# torch.matmul(detected_points,self.transfo_link.T)
		self.ros_topic_manager.pub_message("ground_removal"   ,point_cloud_4d(torch.matmul(ground_plant,self.transfo_link_4d.T).cpu(), self.frame_id_link))
		self.ros_topic_manager.pub_message("index_voxels"     ,point_cloud_4d(torch.matmul(colored_vox,self.transfo_link_4d.T).cpu(), self.frame_id_link))
		self.ros_topic_manager.pub_message("plants_only"      ,point_cloud_3d(torch.matmul(plants_pointcloud,self.transfo_link_3d.T).cpu(), self.frame_id_link))
		self.ros_topic_manager.pub_message("ground_only"      ,point_cloud_3d(torch.matmul(ground_pointcloud,self.transfo_link_3d.T).cpu(), self.frame_id_link))
		self.ros_topic_manager.pub_message("plan_approximed"  ,point_cloud_3d(torch.matmul(plan_approximed,self.transfo_link_3d.T).cpu(), self.frame_id_link))
		self.ros_topic_manager.pub_message("local_max_plants" ,point_cloud_4d(torch.matmul(local_max_plant,self.transfo_link_4d.T).cpu(), self.frame_id_link))
		self.ros_topic_manager.pub_message("density_poi_plant"  ,point_cloud_4d(torch.matmul(density_poi_plant,self.transfo_link_4d.T).cpu(), self.frame_id_link))
		self.ros_topic_manager.pub_message("plants_detected"  ,point_cloud_4d(torch.matmul(plants_detected,self.transfo_link_4d.T).cpu(), self.frame_id_link))
		self.ros_topic_manager.pub_message("density_plants"   ,point_cloud_4d(torch.matmul(density_plant,self.transfo_link_4d.T).cpu(), self.frame_id_link))
		



	def map_callback(self,msg: PointCloud2):
		t0 = time.time()
		pointcloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
		print(pointcloud.shape)

		if len(pointcloud):
			pointcloud = torch.tensor(pointcloud,dtype=torch.float32,device=self.device)
			self.ros_topic_manager.pub_message("pointcloud_received"   ,point_cloud_3d(pointcloud.cpu(), "base_link"))
			self.split_and_compute(pointcloud)
			t1 = time.time()
			if self.full_debug or True:
				print("Total computed in : ",(t1-t0)*1000,"ms")

if __name__ == '__main__':
	rospy.init_node('process_pointcloud')
	ProcessPointcloud()
	rospy.spin()





