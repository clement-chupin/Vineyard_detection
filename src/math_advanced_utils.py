
import torch
from ros_topic_utils import publish_lines


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
    start_x,start_y,end_x,end_y = -6.0,-6.0,6.0,6.0
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
        


def ground_approx_poly_n(pointcloud,ground_approx_error=0.1,order=2,init_abc=None): #speed_speed
    if init_abc is None:
        abc_outliers = plan_approx_poly_n(pointcloud,order)
    else:
        abc_outliers = init_abc
    error 	     = error_of_approx_poly_n(pointcloud,abc_outliers,order)
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



def count_neighbhoroud(poi,pointcloud,r=0.1):

    distance = torch.cdist(poi[:,:2],pointcloud[:,:2])
    nb_neighboorood = ((distance <r)*1.0).sum(-1)  
    return nb_neighboorood



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
        
    #publish_lines()

    return foot_selected









#########    OLD FUNC     ###################

def grid_based_on_param(abc,device):
    nb_x,nb_y = 40,40
    start_x,start_y,end_x,end_y = -6.0,-6.0,6.0,6.0
    range_x,range_y = end_x-start_x,end_y-start_y
    x,y = torch.arange(start_x,end_x,range_x/nb_x,device=device),torch.arange(start_y,end_y,range_y/nb_y,device=device)
    xs,ys = torch.meshgrid(x,y)
    XY = torch.stack([xs.view(nb_x,nb_y),ys.view(nb_x,nb_y)],2).view(nb_x*nb_y,2)
    Z = XY[:,0]*abc[0] + XY[:,1]*abc[1] + abc[2]
    return torch.hstack([XY,Z.view(-1,1)])

def grid_based_on_param_poly(abc,device):
    nb_x,nb_y = 40,40
    start_x,start_y,end_x,end_y = -6.0,-6.0,6.0,6.0
    range_x,range_y = end_x-start_x,end_y-start_y
    x,y = torch.arange(start_x,end_x,range_x/nb_x,device=device),torch.arange(start_y,end_y,range_y/nb_y,device=device)
    xs,ys = torch.meshgrid(x,y)

    XY1 = torch.stack([xs.view(nb_x,nb_y),ys.view(nb_x,nb_y)],2).view(nb_x*nb_y,2)
    XY2 = XY1**2
    XY3 = XY1**3
    Z = (
        XY3[:,0]*abc[0] + XY3[:,1]*abc[1]+
        XY2[:,0]*abc[2] + XY2[:,1]*abc[3]+
        XY1[:,0]*abc[4] + XY1[:,1]*abc[5]+
        abc[6])
    return torch.hstack([XY1,Z.view(-1,1)])




def plan_approx(pointcloud):
    M = torch.hstack([
        pointcloud[:,:2],
        torch.ones((pointcloud.shape[0],1),device=pointcloud.device)
    ])
    V = pointcloud[:,2].view(-1,1)
    O = torch.matmul(torch.linalg.pinv(M),V)
    return O.reshape(-1)

def plan_approx_poly(pointcloud):
    M = torch.hstack([
        pointcloud[:,:2]**3,
        pointcloud[:,:2]**2,
        pointcloud[:,:2]**1,
        torch.ones((pointcloud.shape[0],1),device=pointcloud.device)
    ])
    V = pointcloud[:,2].view(-1,1)
    O = torch.matmul(torch.linalg.pinv(M),V)
    return O.reshape(-1)


def error_of_approx_poly(pointcloud,abc):
    XY3 = pointcloud[:,:2]**3
    XY2 = pointcloud[:,:2]**2
    XY1 = pointcloud[:,:2]**1

    Z_approx = (
        XY3[:,0]*abc[0] + XY3[:,1]*abc[1]+
        XY2[:,0]*abc[2] + XY2[:,1]*abc[3]+
        XY1[:,0]*abc[4] + XY1[:,1]*abc[5]+
        abc[6])

    return pointcloud[:,2]-Z_approx

def error_of_approx(pointcloud,abc):
    XY = pointcloud[:,:2]
    Z_approx = XY[:,0]*abc[0] + XY[:,1]*abc[1] + abc[2]
    return pointcloud[:,2]-Z_approx

def ground_approx_poly(pointcloud,ground_approx_error=0.1): #speed_speed
    abc_outliers = plan_approx_poly(pointcloud)
    error 	     = error_of_approx_poly(pointcloud,abc_outliers)
    abc_inliers  = plan_approx_poly(pointcloud[error < ground_approx_error])
    return abc_inliers




def ground_approx(pointcloud,ground_approx_error=0.1): #speed_speed
    abc_outliers = plan_approx(pointcloud)
    error 	     = error_of_approx(pointcloud,abc_outliers)
    abc_inliers  = plan_approx(pointcloud[error < ground_approx_error])
    return abc_inliers




def keep_pointcloud_in_plan_interval(pointcloud,params,interval):
    e = error_of_approx(pointcloud,params)
    cond_a = e>interval[0]
    cond_b = e<interval[1]
    return pointcloud[cond_a*cond_b]

def keep_pointcloud_in_plan_interval_poly(pointcloud,params,interval):
    e = error_of_approx_poly(pointcloud,params)
    cond_a = e>interval[0]
    cond_b = e<interval[1]
    return pointcloud[cond_a*cond_b]

def proj_pc_on_plan(pointcloud,abc):
    X = pointcloud[:,0]#/(1+torch.abs(abc[0]))
    Y = pointcloud[:,1]#/(1+torch.abs(abc[1]))
    Z_approx = X*abc[0] + Y*abc[1] + abc[2]
    Z = torch.abs(pointcloud[:,2]-Z_approx)
    return torch.hstack([X.view(-1,1),Y.view(-1,1),Z.view(-1,1)])
