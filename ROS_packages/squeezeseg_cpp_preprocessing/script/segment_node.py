#!/usr/bin/python2
# -*- coding: utf-8 -*-
'''
   #+DESCRITION:  online segmentation
   #+FROM:        github.com/durant35/SqueezeSeg
   #+DATE:        2018-08-08-Wed
   #+AUTHOR:      Edward Im (edwardim@snu.ac.kr)
'''
import sys
import os.path
import numpy as np
from PIL import Image

import tensorflow as tf

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header
from std_msgs.msg import Int8

sys.path.append("/home/dyros-vehicle/gitrepo/ims_ros/catkin_ws_kinetic/src/squeezeseg_cpp_preprocessing/script/squeezeseg")
sys.path.append("./squeezeseg")
from config import *
from nets import SqueezeSeg
from utils.util import *
from utils.clock import Clock

from imdb import kitti # ed: header added

def _make_point_field(num_field):
    msg_pf1 = pc2.PointField()
    msg_pf1.name = np.str('x')
    msg_pf1.offset = np.uint32(0)
    msg_pf1.datatype = np.uint8(7)
    msg_pf1.count = np.uint32(1)

    msg_pf2 = pc2.PointField()
    msg_pf2.name = np.str('y')
    msg_pf2.offset = np.uint32(4)
    msg_pf2.datatype = np.uint8(7)
    msg_pf2.count = np.uint32(1)

    msg_pf3 = pc2.PointField()
    msg_pf3.name = np.str('z')
    msg_pf3.offset = np.uint32(8)
    msg_pf3.datatype = np.uint8(7)
    msg_pf3.count = np.uint32(1)

    msg_pf4 = pc2.PointField()
    msg_pf4.name = np.str('intensity')
    msg_pf4.offset = np.uint32(16)
    msg_pf4.datatype = np.uint8(7)
    msg_pf4.count = np.uint32(1)

    if num_field == 4:
        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]

    msg_pf5 = pc2.PointField()
    msg_pf5.name = np.str('label')
    msg_pf5.offset = np.uint32(20)
    msg_pf5.datatype = np.uint8(4)
    msg_pf5.count = np.uint32(1)

    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]


class SegmentNode():
    """LiDAR point cloud segment ros node"""

    def __init__(self,
                 sub_topic, pub_topic, FLAGS):
        # os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

        self._mc = kitti_squeezeSeg_config()
        self._mc.LOAD_PRETRAINED_MODEL = False
        self._mc.BATCH_SIZE = 1         # TODO(bichen): fix this hard-coded batch size.
        self._model = SqueezeSeg(self._mc)
        self._saver = tf.train.Saver(self._model.model_params)

        self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self._saver.restore(self._session, FLAGS.checkpoint)

        self._sub = rospy.Subscriber("/ss_filtered", PointCloud2, self.point_cloud_callback, queue_size=1)
        self._pub = rospy.Publisher(pub_topic, PointCloud2, queue_size=1)

        rospy.spin()

    def point_cloud_callback(self, cloud_msg):
        """

        :param cloud_msg:
        :return:
        """
        clock = Clock()
        # rospy.logwarn("subscribed. width: %d, height: %u, point_step: %d, row_step: %d",
        #               cloud_msg.width, cloud_msg.height, cloud_msg.point_step, cloud_msg.row_step)

        pc = pc2.read_points(cloud_msg, skip_nans=False, field_names=("x", "y", "z","intensity","d"))
        # to conver pc into numpy.ndarray format
        np_p = np.array(list(pc))

        # print("shape : {}".format(np_p.shape))

        # get depth map
        lidar = np_p.reshape(64,512,5)

        # print("{}".format(lidar.shape))

        lidar_f = lidar.astype(np.float32)

        # to perform prediction
        lidar_mask = np.reshape(
            (lidar[:, :, 4] > 0),
            [self._mc.ZENITH_LEVEL, self._mc.AZIMUTH_LEVEL, 1]
        )
        lidar_f = (lidar_f - self._mc.INPUT_MEAN) / self._mc.INPUT_STD
        pred_cls = self._session.run(
            self._model.pred_cls,
            feed_dict={
                self._model.lidar_input: [lidar_f],
                self._model.keep_prob: 1.0,
                self._model.lidar_mask: [lidar_mask]
            }
        )
        label = pred_cls[0]

        ## point cloud for SqueezeSeg segments
        x = lidar[:, :, 0].reshape(-1)
        y = lidar[:, :, 1].reshape(-1)
        z = lidar[:, :, 2].reshape(-1)
        i = lidar[:, :, 3].reshape(-1)
        label = label.reshape(-1)
        cloud = np.stack((x, y, z, i, label))


        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = "velodyne_link"

        # point cloud segments
        # 4 PointFields as channel description
        msg_segment = pc2.create_cloud(header=header,
                                       fields=_make_point_field(cloud.shape[0]),
                                       points=cloud.T)

        # ed: /squeeze_seg/points publish
        self._pub.publish(msg_segment)
        rospy.loginfo("Point cloud processed. Took %.6f ms.", clock.takeRealTime())
