/*
 * #+DESCRIPTION: c++ conversion of preprocessor from velodyne points to [512,64] dataset
 *                strongly inspried from github.com/durant35/SqueezeSeg
 *
 * #+DATE:        2018-08-28 Tue
 * #+AUTHOR:      Edward Im (edwardim@snu.ac.kr)
 */
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Int8.h>

#define PCL_NO_PRECOMPILE
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/conditional_removal.h>

#include <eigen3/Eigen/Dense>

#include <iostream>
#include <cmath>

#define D2R  M_PI/180.0
#define R2D  180.0/M_PI

using namespace std;

namespace pcl {
struct PointXYZID {
    PCL_ADD_POINT4D                     // Macro quad-word XYZ
    float intensity;                    // Laser intensity
    float d;                            // Distance for SqueezeSeg 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // Ensure proper alignment
} EIGEN_ALIGN16;
} //end of namespace pcl

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZID,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (float, d, d)
                                  )
PCL_INSTANTIATE_PointCloud(pcl::PointXYZID);

typedef pcl::PointXYZI VPoint;
typedef pcl::PointCloud<VPoint> VPointCloud;
typedef pcl::PointXYZID VPointXYZID;
typedef pcl::PointCloud<VPointXYZID> VPointCloudXYZID;

ros::Publisher ss_pub;

// /velodyne_points topic's subcriber callback function
void velo_callback(const sensor_msgs::PointCloud2::ConstPtr &msg){
  VPointCloud::Ptr pcl_msg(new VPointCloud);
  pcl::fromROSMsg(*msg, *pcl_msg);

  if(pcl_msg->points.empty())
    return;

  VPointCloud::Ptr out_msg(new VPointCloud);
  out_msg->points.clear();

  // hv_in_range substitute loop
  for(int i=0; i <= pcl_msg->points.size()-1; i++) {
    double x = pcl_msg->points[i].x;
    double y = pcl_msg->points[i].y;
    double z = pcl_msg->points[i].z;

	double a = atan2(y,x) * R2D;
	if(a > -45 && a < 45) 
		out_msg->points.push_back(pcl_msg->points[i]);
  }

  VPointCloudXYZID filtered_cloud;
  sensor_msgs::PointCloud2 ss_msg;
  filtered_cloud.points.resize(32768);  // 64 * 512

  double dphi = 180/512.0 * D2R;
  double dtheta = 0.4 * D2R;

  // pto_depth_map substitute loop
  for(int i=0; i <= out_msg->points.size(); i++) {
    double x = out_msg->points[i].x;
    double y = out_msg->points[i].y;
    double z = out_msg->points[i].z;
    double d = sqrt(x*x + y*y + z*z);
    double r = sqrt(x*x + y*y);

    double phi = 90 * D2R  - asin(y/r);
    double phi2 = (int)(phi/dphi);
    double theta = 2 * D2R - asin(z/d);
    double theta2 = (int)(theta/dtheta);

    if(phi2 < 0)
      phi2 = 0;
    if(phi2 >= 512)
      phi2 = 511;

    if(theta2 < 0)
      theta2 = 0;
    if(theta2 >= 64)
      theta2 = 63;

    filtered_cloud.points[theta2*512 + phi2].x = out_msg->points[i].x;
    filtered_cloud.points[theta2*512 + phi2].y = out_msg->points[i].y;
    filtered_cloud.points[theta2*512 + phi2].z = out_msg->points[i].z;
    // TODO(edward): SqueezeSeg got error if there are intensity data from here. need to fix
    /* filtered_cloud.points[theta2*512 + phi2].intensity = out_msg->points[i].intensity; */
    filtered_cloud.points[theta2*512 + phi2].d = d;
  }
  // cout << "[+] " << filtered_cloud.points.size() << endl;  // DEBUG

  pcl::toROSMsg(filtered_cloud, ss_msg);
  ss_msg.header.frame_id = "velodyne_link";
  ss_msg.header.stamp = ros::Time::now();

  // publish to /ss_filtered
  ss_pub.publish(ss_msg);
  filtered_cloud.points.clear();
}

int main(int argc, char **argv){
  ros::init(argc, argv, "cpp_preprocessing");

  ros::NodeHandle nh;

  ros::Subscriber velodyne_sub = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 1, velo_callback);

  ss_pub = nh.advertise<sensor_msgs::PointCloud2>("/ss_filtered",1);

  ros::spin();
  return 0;
}
