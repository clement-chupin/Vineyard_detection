# Vineyard_detection
Detect the vineyards foot, based on LiDAR or depth-camera, processed in real time, based on pytorch

## Demo :

![](https://github.com/clement-chupin/Vineyard_detection/blob/master/images/demo_2.gif)
![](https://github.com/clement-chupin/Vineyard_detection/blob/master/images/foot_detection_shema.png)



## Install :

`conda env create -f environment.yml`

or

`pip install numpy==1.20 torch==2.0.0 rosnumpy`

## Source your env :

Ros-noetic :

`source /opt/ros/noetic/setup.bash`

Python environement :

`conda activate py_ros`

## Run :

`roslaunch Vineyard_detection stephane_cam_z.launch`

or

`roslaunch NEW_ROS_PACKET_NAME stephane_cam_z.launch`

## Doc :

https://github.com/clement-chupin/Vineyard_detection/blob/master/doc/Rapport_inrae_M2.pdf
