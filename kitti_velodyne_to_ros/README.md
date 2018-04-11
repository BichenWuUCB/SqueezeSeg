### .npy file visualizer for ROS
---
- basically It has two commands 
- one is for `.bin` file from KITTI velodyne datasets, another is for `.npy` file from SqueezeSeg datasets
  - commands are introduced below


<br><br>
![](https://github.com/tigerk0430/SqueezeSeg/blob/master/kitti_velodyne_to_ros/gif_and_pics/squeezeseg_npy.gif)


<br><br>
### Instrunction
---
1. add codes in your `eval.py` like below 117-122 lines
  1.1 then you can save `pred_~~~.npy` files after evaluation. It'd be prediction data
![](https://github.com/tigerk0430/SqueezeSeg/blob/master/kitti_velodyne_to_ros/gif_and_pics/added_code.png)
2. run `eval.sh` and save `pred_~~~.npy` files into specific path  
3. move this `kitty_velodyne_to_ros` folder to your `catkin_ws` folder and run `catkin_make`
4. modify `.launch` file to your data paths.
5. run commands below


<br><br>
### Available commands
--- 
(for `.bin` file from KITTI datasets)
1. `rosrun kitti_velodyne_to_ros kitti_velodyne_to_ros_node`

(for `.npy` file from SqueezeSeg) (after modifying path in .launch file)

2. `roslaunch kitti_velodyne_to_ros npy_velodyne_to_ros.launch`
