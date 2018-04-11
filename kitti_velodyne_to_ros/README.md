### .npy file visualizer for ROS
---

![](https://github.com/tigerk0430/SqueezeSeg/blob/master/kitti_velodyne_to_ros/gif_and_pics/squeezeseg_npy.gif)


### instrunction
---
1. move this file to your `catkin_ws` folder and run `catkin_make`
2. modify `.launch` file to your data paths.
3. run commands below



### available commands
--- 
(for `.bin` file from KITTI datasets)
1. `rosrun kitti_velodyne_to_ros kitti_velodyne_to_ros_node`

(for `.npy` file from SqueezeSeg) (after modifying path in .launch file)

2. `roslaunch kitti_velodyne_to_ros npy_velodyne_to_ros.launch`
