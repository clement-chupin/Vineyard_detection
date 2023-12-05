#!/usr/bin/env python3
import sys
import os
import time

import numpy as np
import ros_numpy
import torch
import torch.nn as nn
import rospy

from sensor_msgs.msg import PointCloud2
from rospy.numpy_msg import numpy_msg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ros_topic_utils import point_cloud_3d
from ros_topic_utils import point_cloud_4d
from ros_topic_utils import born_lidar_range
from ros_topic_utils import bord_lidar_height




from octotree_utils import OctreeTree
from octotree_utils import LocalMaxSearching

from ros_topic_utils import ros_topic_manager
from math_advanced_utils import grid_based_on_param_poly_n
from math_advanced_utils import ground_approx_poly_n
from math_advanced_utils import keep_pointcloud_in_plan_interval_poly_n

from math_advanced_utils import proj_pc_on_plan_poly_n
from math_advanced_utils import merge_by_distance
from math_advanced_utils import count_neighbourhood

from multiprocessing.pool import ThreadPool as Pool

class ProcessPointcloud:
	def __init__(self):

		self.activate                 = rospy.get_param('~activate', True)
		self.use_gpu                  = rospy.get_param('~use_gpu', True)
		self.debug                    = rospy.get_param('~debug', True)

		self.frame_id_link            = rospy.get_param('~frame_id_link',"velo_link")
		self.transfo_link = rospy.get_param('~transfo_link', [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
		
		self.range_lidar_min          = rospy.get_param('~lidar_range_min', 0.0)
		self.range_lidar_max          = rospy.get_param('~lidar_range_max', 100.0)
		self.z_lidar_min_limit		  = rospy.get_param('~lidar_z_min', -2.0)
		self.z_lidar_max_limit		  = rospy.get_param('~lidar_z_max', 4.0)

		self.octo_npoints_max         = rospy.get_param('~octo_npoints_max', 1000)
		self.octo_depth_max           = rospy.get_param('~octo_depth_max', 100)
		self.octo_padding_size        = rospy.get_param('~octo_padding_size', 0.05)

		self.voxel_ground_size        = rospy.get_param('~ground_segmentation_voxel_size', 0.1)
		self.ground_segmentation_a    = rospy.get_param('~ground_segmentation_a', -0.6) #unused
		self.ground_segmentation_b    = rospy.get_param('~ground_segmentation_b', 0.05) #unused
		
		self.density_max_a            = rospy.get_param('~density_max_a', -100)
		self.density_max_b            = rospy.get_param('~density_max_b', 0.0)
		self.density_kernel_size      = rospy.get_param('~density_kernel_size', 0.1)

		self.final_vine_selection_a   = rospy.get_param('~final_vine_selection_a', -0.5)
		self.final_vine_selection_b   = rospy.get_param('~final_vine_selection_b', 0.0)

		self.ground_approx_order      = rospy.get_param('~ground_approx_order', 3)
		self.ground_approx_error      = rospy.get_param('~ground_approx_error', 0.2)
		self.ground_approx_memory     = rospy.get_param('~ground_approx_memory', False)
		self.final_merging_distance   = rospy.get_param('~final_merging_distance', 0.1)

		self.height_selection_low = rospy.get_param('~height_selection_low', 0.0)
		self.height_selection_mid = rospy.get_param('~height_selection_mid', 0.4)
		self.height_selection_hig = rospy.get_param('~height_selection_hig', 1.2)


		self.ground_segmentator = LocalMaxSearching(-self.ground_segmentation_a,self.ground_segmentation_b,)

		self.threshold_nb_plant_foot_detection = rospy.get_param('~threshold_nb_plant_foot_detection', 4 )

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
		self.plants_poi_octree_kwargs = self.pointcloud_octree_kwargs
		#if you want to modify specific process inside the octree
		self.plants_poi_octree_kwargs["density_kernel_size"] = rospy.get_param('~plant_density_kernel_size', 0.3)

		self.pred_ground_params = None
		self.ros_topic_manager = ros_topic_manager()
	
		self.final_point_selection = LocalMaxSearching(self.final_vine_selection_a,self.final_vine_selection_b)

		if self.activate:
			torch.set_grad_enabled(False)
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
			

	def compute_pointcloud(self,pointcloud):
		pointcloud = born_lidar_range(pointcloud,self.range_lidar_min,self.range_lidar_max)
		
		self.ros_topic_manager.pub_message("pointcloud_range_limited",point_cloud_3d(
			torch.matmul(pointcloud,self.transfo_link_3d.T).cpu(), self.frame_id_link))

		
		ground_params,mini_points = ground_approx_poly_n(
			pointcloud,self.ground_approx_error,self.ground_approx_order,self.voxel_ground_size,self.ground_segmentator)
		self.ros_topic_manager.pub_message("ground_mini_points",point_cloud_3d(
			torch.matmul(mini_points,self.transfo_link_3d.T).cpu(), self.frame_id_link))
		

		visu_plan_approximed = grid_based_on_param_poly_n(ground_params,pointcloud.device,self.ground_approx_order)
		self.ros_topic_manager.pub_message("ground_plan_approximed",point_cloud_3d(
			torch.matmul(visu_plan_approximed,self.transfo_link_3d.T).cpu(), self.frame_id_link))

		ground  = keep_pointcloud_in_plan_interval_poly_n(pointcloud,ground_params,
													[-1e10,self.height_selection_low],self.ground_approx_order)
		plant_in_foot_area  = keep_pointcloud_in_plan_interval_poly_n(pointcloud,ground_params,
																[self.height_selection_low,self.height_selection_mid],self.ground_approx_order)
		plant_in_plant_area = keep_pointcloud_in_plan_interval_poly_n(pointcloud,ground_params,
																[self.height_selection_mid,self.height_selection_hig],self.ground_approx_order)
		
		self.ros_topic_manager.pub_message("ground_pointcloud",point_cloud_3d(
			torch.matmul(ground,self.transfo_link_3d.T).cpu(), self.frame_id_link))
		self.ros_topic_manager.pub_message("foot_pointcloud",point_cloud_3d(
			torch.matmul(plant_in_foot_area,self.transfo_link_3d.T).cpu(), self.frame_id_link))
		self.ros_topic_manager.pub_message("plant_pointcloud",point_cloud_3d(
			torch.matmul(plant_in_plant_area,self.transfo_link_3d.T).cpu(), self.frame_id_link))
		
		if len(plant_in_foot_area) !=0:
			foot_area_octree = OctreeTree(plant_in_foot_area,self.pointcloud_octree_kwargs)
			
			foot_area     		= foot_area_octree.compute("get_entier_pc")
			unused_density_foot = foot_area_octree.compute("get_density")
			local_max_foot_area = foot_area_octree.compute("get_local_max")
			foot_area_poi  = foot_area[local_max_foot_area[:,3]==1]

			final_foot_selected = merge_by_distance(foot_area_poi,self.final_merging_distance)
			final_foot_selected = proj_pc_on_plan_poly_n(final_foot_selected,ground_params,self.ground_approx_order)
			self.ros_topic_manager.pub_message("foot_detected",point_cloud_3d(torch.matmul(
				final_foot_selected,self.transfo_link_3d.T).cpu(), self.frame_id_link))
			
			if len(plant_in_plant_area) !=0:
				plant_area_octree = OctreeTree(plant_in_plant_area,self.pointcloud_octree_kwargs)
				plant_selected_pointcloud     = plant_area_octree.compute("get_entier_pc")
				unused_density_plant_selected = plant_area_octree.compute("get_density")
				local_max_plant_selected      = plant_area_octree.compute("get_local_max")
				plant_selected_poi = plant_selected_pointcloud[local_max_plant_selected[:,3]==1]
				foot_selection = count_neighbourhood(final_foot_selected,plant_selected_poi)

				final_foot_validate = final_foot_selected[foot_selection > self.threshold_nb_plant_foot_detection]
			
				self.ros_topic_manager.pub_message("foot_validate",point_cloud_3d(
					torch.matmul(final_foot_validate,self.transfo_link_3d.T).cpu(), self.frame_id_link))

	def map_callback(self,msg: PointCloud2):
		t0 = time.time()
		pointcloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
		if len(pointcloud):
			pointcloud = torch.tensor(pointcloud,dtype=torch.float32,device=self.device)
			self.ros_topic_manager.pub_message("raw_pointcloud_received",point_cloud_3d(pointcloud.cpu(), "base_link"))
			self.compute_pointcloud(pointcloud)
			t1 = time.time()
			if self.debug:
				print(pointcloud.shape[0],"points")
				print(round(1/(t1-t0)),"Hz")
				print(round((t1-t0)*1000),"ms")

if __name__ == '__main__':
	rospy.init_node('process_pointcloud')
	ProcessPointcloud()
	rospy.spin()




