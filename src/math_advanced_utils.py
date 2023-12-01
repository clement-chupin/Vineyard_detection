
import torch
from ros_topic_utils import publish_lines
from multiprocessing.pool import ThreadPool as Pool
import time


def pca(pointcloud):
	centered_pc = pointcloud-pointcloud.mean(0)
	cov = torch.matmul(centered_pc.T,centered_pc)
	eigenvalues, eigenvectors = torch.linalg.eigh(cov)
	return eigenvectors[:, torch.argmax(eigenvalues)]


def plan_approx_poly_n(pointcloud,order=2):
	all_XY = []
	for o in range(order):
		all_XY.append(pointcloud[:,:2]**(o+1))
	all_XY.append(torch.ones((pointcloud.shape[0],1),device=pointcloud.device))

	M = torch.hstack(all_XY)
	V = pointcloud[:,2].view(-1,1)
	O = torch.matmul(torch.linalg.pinv(M),V)
	return O.reshape(-1)



def grid_based_on_param_poly_n(abc,device,order=2):
	nb_x,nb_y = 40,40
	start_x,start_y,end_x,end_y = -3.0,-6.0,10.0,6.0
	range_x,range_y = end_x-start_x,end_y-start_y
	x,y = torch.arange(start_x,end_x,range_x/nb_x,device=device),torch.arange(start_y,end_y,range_y/nb_y,device=device)
	xs,ys = torch.meshgrid(x,y)
	XY = torch.stack([xs.view(nb_x,nb_y),ys.view(nb_x,nb_y)],2).view(nb_x*nb_y,2)
	Z_approx = torch.ones((XY.shape[0],),device=XY.device)*abc[-1]
	for o in range(order):
		XY_o = XY**(o+1)
		Z_approx += (XY_o*abc[o*2:o*2+2].view(1,-1)).sum(-1)
	return torch.hstack([XY,Z_approx.view(-1,1)])


def error_of_approx_poly_n(pointcloud,abc,order=2):
	Z_approx = torch.ones((pointcloud.shape[0],),device=pointcloud.device)*abc[-1]
	for o in range(order):
		XY = pointcloud[:,:2]**(o+1)
		Z_approx += (XY*abc[o*2:o*2+2].view(1,-1)).sum(-1)
	return pointcloud[:,2]-Z_approx

def proj_pc_on_plan_poly_n(pointcloud,abc,order=2):
	Z = torch.ones((pointcloud.shape[0],),device=pointcloud.device)*abc[-1]
	for o in range(order):
		XY = pointcloud[:,:2]**(o+1)
		Z += (XY*abc[o*2:o*2+2].view(1,-1)).sum(-1)
	return torch.hstack([pointcloud[:,:2],Z.view(-1,1)])




def unique_optimized(tensor,dim=0):
	n=1000000
	if len(tensor) > n:
		return torch.vstack([
			unique_optimized(tensor[:n],dim=dim),
			unique_optimized(tensor[n:],dim=dim)])
	else:
		return torch.unique(tensor,dim=dim)

def get_poincloud_by_tile_index(index_tiles,i_tile,pointcloud):
	return pointcloud[torch.all((index_tiles[:,:2] == i_tile),dim=-1)]

def get_min_poincloud_by_tile_index(index_tiles,i_tile,pointcloud):
	return pointcloud[torch.all((index_tiles[:,:2] == i_tile),dim=-1),2].min()



def points_to_voxel_grid(points, voxel_size):
	min_coords = points.min(dim=0).values
	max_coords = points.max(dim=0).values
	grid_dims = ((max_coords - min_coords) / voxel_size).ceil().to(torch.int64)
	voxel_grid = torch.zeros((grid_dims[0], grid_dims[1], grid_dims[2]), dtype=torch.bool,device=points.device)
	voxel_indices = ((points - min_coords) / voxel_size).to(torch.int64)
	voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
	return voxel_grid


def grille_XYZ(Nx,Ny,Nz,device):
	x,y,z = torch.arange(0,Nx,1,device=device),torch.arange(0,Ny,1,device=device),torch.arange(0,Nz,1,device=device),
	xs,ys,zs = torch.meshgrid(x,y,z)
	XYZ = torch.stack([
		xs.view(Nx,Ny,Nz),
		ys.view(Nx,Ny,Nz),
		zs.view(Nx,Ny,Nz)
		],3)#.view(Nx*Ny*Nz,3)
	return XYZ

def grille_XY(Nx,Ny,device):
	x,y, = torch.arange(0,Nx,1,device=device),torch.arange(0,Ny,1,device=device)
	xs,ys = torch.meshgrid(x,y)
	XY = torch.stack([
		xs.view(Nx,Ny),
		ys.view(Nx,Ny),
		],2).view(Nx*Ny,2)
	return XY

def lowest_voxels_per_column_parallel(voxel_grid):
	Nx,Ny,Nz = voxel_grid.shape
	grille = torch.arange(0,voxel_grid.shape[2],1,device=voxel_grid.device).view(1,1,-1).repeat(voxel_grid.shape[0],voxel_grid.shape[1],1)
	transposed_grid = (grille*1.0*voxel_grid) + (1-voxel_grid*1)*1e7
	low_lined = transposed_grid.min(-1).values.view(Nx*Ny)
	return torch.hstack([grille_XY(Nx,Ny,voxel_grid.device),low_lined.view(-1,1)])[low_lined!=1e7]
	

def keep_min_in_voxel_grid(pointcloud,voxel_size):
	min_coords = pointcloud.min(dim=0).values
	voxel_grid = points_to_voxel_grid(pointcloud,voxel_size)
	low = lowest_voxels_per_column_parallel(voxel_grid)
	return low*voxel_size + min_coords.view(1,3)

def ground_approx_poly_n(pointcloud,ground_approx_error=0.1,order=2,init_abc=None,voxel_size=0.1): #speed_speed
	if init_abc is None:
		abc_outliers = plan_approx_poly_n(keep_min_in_voxel_grid(pointcloud,voxel_size),1)
	else:
		abc_outliers = init_abc
	error 	     = torch.abs(error_of_approx_poly_n(pointcloud,abc_outliers,1))
	abc_inliers  = plan_approx_poly_n(pointcloud[error < ground_approx_error],order)
	return abc_inliers


def keep_pointcloud_in_plan_interval_poly_n(pointcloud,params,interval,order=2):
	e = error_of_approx_poly_n(pointcloud,params,order)
	cond_a = e>interval[0]
	cond_b = e<interval[1]
	return pointcloud[cond_a*cond_b]



def merge_by_distance(pointcloud,final_merging_distance=0.2):
	distance = torch.cdist(pointcloud,pointcloud)
	is_neighboorood = (distance <final_merging_distance)*1.0
	cluster = torch.matmul(is_neighboorood,is_neighboorood)
	cluster = torch.matmul(cluster,cluster)
	cluster = torch.matmul(cluster,cluster)
	coords = pointcloud.view(1,pointcloud.shape[0],pointcloud.shape[1]).repeat(pointcloud.shape[0],1,1)
	clust_transfo =  (cluster.bool()*1).view(cluster.shape[0],cluster.shape[1],1)
	selected_coord = coords*clust_transfo
	means = selected_coord.sum(1) /((selected_coord.bool()*1).sum(1)+1e-9)
	sigma_points = torch.unique(means,dim=0)
	# print(sigma_points)
	return sigma_points

def count_neighbourhood(poi,pointcloud,r=0.1):

	distance = torch.cdist(poi[:,:2],pointcloud[:,:2])
	nb_neighboorood = ((distance <r)*1.0).sum(-1)  
	return nb_neighboorood


### unused functions :
def foot_selection(self,foot,poi,r=0.1,height_threshold=0.3,marker_link="base_link"):#P,N
	F = foot.shape[0]
	N = poi.shape[0]
	D = foot.shape[-1]
	distance_poi_poi = torch.cdist(poi,poi)    #N,N,
	poi_dupli = poi.view(1,N,D).repeat(N,1,1)  #N,N,D
	foot_dupli= foot.view(F,1,D).repeat(1,N,1) #F,N,D
	adj_poi_poi = (distance_poi_poi < r)*1.0 #N,N
	distance_foot_poi = torch.cdist(foot,poi)  #F,N
	cluster_foot = (distance_foot_poi < r).view(F,N,1)   #F,N,1
	cluster_poi  = torch.matmul(adj_poi_poi,adj_poi_poi) #N,N
	cluster_poi  = torch.matmul(cluster_poi,cluster_poi) #N,N
	cluster_poi  = torch.matmul(cluster_poi,cluster_poi).bool().view(N,N,1)#N,N,1
	cluster_and_coord_poi_poi  = poi_dupli*(cluster_poi*1.0)    #N,N,D
	cluster_and_coord_poi_foot = foot_dupli*(cluster_foot*1.0)  #F,N,D
	max_poi_connected_height = cluster_and_coord_poi_poi[:,:,-1].max(-1).values #N,
	max_poi_dupli_to_foot = max_poi_connected_height.view(1,N).repeat(F,1,) #F,N
	dist_foot_to_poi = max_poi_dupli_to_foot-cluster_and_coord_poi_foot[:,:,-1]  #F,N
	max_dist = (dist_foot_to_poi*cluster_foot[:,:,0]).max(1).values #F
	foot_selected = max_dist>height_threshold #F

	def plot_cluster(cluster_ab,points_a,points_b):
		ind = torch.argwhere(cluster_ab)
		list_A = []
		list_B = []
		for i in ind:
			a = points_a[i[0]]
			b = points_b[i[1]]
			list_A.append(a)
			list_B.append(b)
		publish_lines(list_A,list_B,marker_link)
	N_reduce = 200
	poi_trans = torch.matmul(poi,self.transfo_link_3d.T)
	foot_trans = torch.matmul(foot,self.transfo_link_3d.T)


	# print(cluster_poi.view(N,N)*1)
	# plot_cluster(cluster_poi.view(N,N)[:N_reduce,:N_reduce],poi_trans,poi_trans)

	print(cluster_foot.view(F,N)*1)
	plot_cluster(cluster_foot.view(F,N)[:N_reduce,:N_reduce],foot_trans,poi_trans)

	return foot_selected



