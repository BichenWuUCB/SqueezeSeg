#!/usr/bin/python2
# -*- coding: utf-8 -*-
'''
   #+DESCRITION:  online segmentation
   #+FROM:        github.com/durant35/SqueezeSeg
   #+DATE:        2018-07-23-Mon
   #+AUTHOR:      Edward Im (edwardim@snu.ac.kr)
'''
import argparse
import tensorflow as tf
import rospy

from segment_node import SegmentNode

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint', '/home/dyros-vehicle/gitrepo/ims_ros/catkin_ws_kinetic/src/squeezeseg_cpp_preprocessing/checkpoint/SqueezeSeg/model.ckpt-23000',
    """Path to the model parameter file.""")

if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser(description='LiDAR point cloud semantic segmentation')
    parser.add_argument('--sub_topic', type=str,
                        help='the pointcloud message topic to be subscribed, default `/ss_filtered`',
                        default='/ss_filtered')
    parser.add_argument('--pub_topic', type=str,
                        help='the pointcloud message topic to be published, default `/squeeze_seg/points`',
                        default='/squeeze_seg/points')

    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('segment_node')

    node = SegmentNode(sub_topic=args.sub_topic,
                       pub_topic=args.pub_topic,
                       FLAGS=FLAGS)

