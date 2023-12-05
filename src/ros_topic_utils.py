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
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def point_cloud_4d(points, parent_frame):
	""" Creates a point cloud message.
	Args:
		points: Nx4 array of xyz positions (m) and intensity (0..1)
		parent_frame: frame in which the point cloud is defined
	Returns:
		sensor_msgs/PointCloud2 message
	"""
	points = np.array(points.cpu())
	ros_dtype = sensor_msgs.PointField.FLOAT32
	dtype = np.float32
	itemsize = np.dtype(dtype).itemsize

	data = points.astype(dtype).tobytes()

	fields = [sensor_msgs.PointField(
		name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
		for i, n in enumerate(["x", "y", "z","intensity"])]

	header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())

	return sensor_msgs.PointCloud2(
		header=header,
		height=1,
		width=points.shape[0],
		is_dense=False,
		is_bigendian=False,
		fields=fields,
		point_step=(itemsize * 4),
		row_step=(itemsize * 4 * points.shape[0]),
		data=data
	)
def point_cloud_3d(points, parent_frame):
	""" Creates a point cloud message.
	Args:
		points: Nx3 array of xyz positions (m)
		parent_frame: frame in which the point cloud is defined
	Returns:
		sensor_msgs/PointCloud2 message
	"""
	points = np.array(points)
	ros_dtype = sensor_msgs.PointField.FLOAT32
	dtype = np.float32
	itemsize = np.dtype(dtype).itemsize

	data = points.astype(dtype).tobytes()

	fields = [sensor_msgs.PointField(
		name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
		for i, n in enumerate(["x", "y", "z"])]

	header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())

	return sensor_msgs.PointCloud2(
		header=header,
		height=1,
		width=points.shape[0],
		is_dense=False,
		is_bigendian=False,
		fields=fields,
		point_step=(itemsize * 3),
		row_step=(itemsize * 3 * points.shape[0]),
		data=data
	)


class ros_topic_manager:
	def __init__(self):
		self.dict_topic = {}

	def init_publisher(self,name,type=PointCloud2):
		publisher = rospy.Publisher(name, type,queue_size=1)
		self.dict_topic.update({name:publisher})

	def pub_message(self,name,message):
		if name not in self.dict_topic:
			self.init_publisher(name)
		self.dict_topic[name].publish(message)
		

def born_lidar_range(pointcloud,born_min,born_max):
	selection_a = torch.sqrt((pointcloud**2).sum(1)) >= born_min
	selection_b = torch.sqrt((pointcloud**2).sum(1)) <= born_max
	return pointcloud[selection_a*selection_b]

def bord_lidar_height(pointcloud,min_limit,max_limit):
	return pointcloud[(pointcloud[:,2] > min_limit)*(pointcloud[:,2] < max_limit)]

















# def get_line(a,b,marker_link="base_link"):
# 	marker = Marker()#zed2i_left_camera_frame
# 	marker.header.frame_id = marker_link
# 	# marker.type = marker.LINE_STRIP
# 	marker.type = 4
# 	marker.action = marker.ADD
# 	marker.ns ="toto"


# 	scale_reduce = 0.8
# 	# marker scale
# 	marker.scale.x = scale_reduce
# 	marker.scale.y = scale_reduce
# 	marker.scale.z = scale_reduce

# 	# marker color
# 	marker.color.a = 1.0
# 	marker.color.r = 1.0
# 	marker.color.g = 0.0
# 	marker.color.b = 0.0

# 	# marker orientaiton
# 	marker.pose.orientation.x = 0.0
# 	marker.pose.orientation.y = 0.0
# 	marker.pose.orientation.z = 0.0
# 	marker.pose.orientation.w = 1.0

# 	# marker position
# 	marker.pose.position.x = a[0].item()
# 	marker.pose.position.y = a[1].item()
# 	marker.pose.position.z = a[2].item()
	

# 	# marker line points
# 	marker.points = []
# 	# first point
# 	first_line_point = Point()
# 	first_line_point.x = a[0].item()
# 	first_line_point.y = a[1].item()
# 	first_line_point.z = a[2].item()

# 	# second point
# 	second_line_point = Point()
# 	second_line_point.x = b[0].item()
# 	second_line_point.y = b[1].item()
# 	second_line_point.z = b[2].item()

# 	marker.points = [first_line_point,second_line_point]

# 	return marker

# def publish_lines(list_A,list_B,marker_link="base_link",marker_array_pub=rospy.Publisher("marker_array", MarkerArray,queue_size=1)):
# 	msg = MarkerArray()
# 	id=0
# 	for idx in range(len(list_A)):
# 		msg.markers.append(get_line(list_A[idx],list_B[idx],marker_link))
# 		msg.markers[-1].id = id
# 		id += 1
# 	# print(msg)
# 	marker_array_pub.publish(msg)

# def send_all_markers(self,publisher):
# 		msg = MarkerArray()
# 		msg.markers = self.all_markers
# 		# for i in range(len(self.all_markers)):
# 		# 	msg.markers.append(self.all_markers[i])
# 		id = 0
# 		for m in msg.markers:
# 			m.id = id
# 			id += 1
# 		publisher.publish(msg)
# 		self.all_markers = []
