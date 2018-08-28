/*
  [+] reference : https://github.com/iralabdisco/kitti_player/blob/public/src/kitti_player.cpp
*/
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>

#include <iostream>
#include <string>
#include <fstream>
#include <dirent.h>


int main(int argc, char **argv) {
  ros::init(argc, argv, "lidar_annotator_node");

  ros::NodeHandle nh;
  std::string path;
  std::string full_file_name;

  nh.param("kitti_bin_path", path,
           std::string("/media/dyros-vehicle/edward_6/datasets/KITTI_datasets/2011_09_26_drive_0005_sync/velodyne_points/data/"));

  ros::Publisher kitti_pub = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points_kitti",1);

  // ed: 5 hz
  ros::Rate loop_rate(5);

  int entries_played = 0;
  int total_entries = 0;

  DIR* dir = opendir(path.c_str());
  struct dirent *ent;
  unsigned int len = 0;

  // ed: get the number of files in directory
  while((ent = readdir(dir))){
    len = strlen(ent->d_name);
    if(len > 2)
      total_entries++;
  }
  closedir(dir);

  /* std::cout << "total entries : " << total_entries << std::endl;  // : 154 */

  while(ros::ok() && entries_played <= total_entries-1) {
    full_file_name = path + boost::str(boost::format("%010d") % entries_played) + ".bin";

    // ed: for debugging
    std::cout << "full file name : " << full_file_name << std::endl;

    // ed: open file
    std::fstream input(full_file_name.c_str(), std::ios::in | std::ios::binary);

    if(!input.good()) {
      ROS_ERROR_STREAM("could not read file: " << full_file_name);
      return 0;
    }
    else {
      ROS_DEBUG_STREAM("reading " << full_file_name);
      // ed: go to beggining of the file
      input.seekg(0, std::ios::beg);

      pcl::PointCloud<pcl::PointXYZI>::Ptr points(new pcl::PointCloud<pcl::PointXYZI>);

      // ed: read data
      for(int i=0; input.good() && !input.eof(); i++) {
        pcl::PointXYZI point;
        input.read((char*)&point.x, 3*sizeof(float));
        input.read((char*)&point.intensity, sizeof(float));
        points->push_back(point);
      }
      input.close();

      sensor_msgs::PointCloud2 pc2;
      pc2.header.frame_id = "velodyne_link";
      pc2.header.stamp = ros::Time::now();
      points->header = pcl_conversions::toPCL(pc2.header);

      // ed: /velodyne_points_kitti publish
      kitti_pub.publish(points);
    }
    loop_rate.sleep();
    entries_played++;
  }

  return 0;
}
