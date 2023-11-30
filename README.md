# Vineyard_detection
Detect the vineyards foot, based on LiDAR or depth-camera, processed in real time, based on pytorch




## Install :

`conda env create -f environment.yml`

or

`python3.8 -m pip install torch torchvision torchaudio`

## Source your env :

Ros-noetic :
`source /opt/ros/noetic/setup.bash`

Python environement :

`conda activate py_ros`

## Run :

`roslaunch Ros_Pointcloud_Real_Time_Analysis stephane_cam_z.launch`
or
`roslaunch NEW_ROS_PACKET_NAME stephane_cam_z.launch`

## Doc :

https://github.com/clement-chupin/Vineyard_detection/blob/master/doc/Rapport_inrae_M2.pdf
