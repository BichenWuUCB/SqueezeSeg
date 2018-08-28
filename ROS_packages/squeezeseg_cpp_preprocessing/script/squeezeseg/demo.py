
#-*- coding: utf-8 -*-
"""Evaluation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import sys
import time
import glob    

import numpy as np
from six.moves import xrange
import tensorflow as tf
from PIL import Image

from config import *
from imdb import kitti
from utils.util import *
from nets import *


# ed: Windows version
IS_WINDOWS=False


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


if IS_WINDOWS:
    # ed: for Windows
    tf.app.flags.DEFINE_string(
            'checkpoint', '.\\data\\SqueezeSeg\\model.ckpt-23000',
            """Path to the model parameter file.""")
    tf.app.flags.DEFINE_string(
            'input_path', '.\\data\\samples\\*',
            """Input lidar scan to be detected. Can process glob input such as """
            """./data/samples/*.npy or single input.""")
    tf.app.flags.DEFINE_string(
            'out_dir', '.\\data\\samples_out\\', """Directory to dump output.""")
else:
    # ed: for Linux
    tf.app.flags.DEFINE_string(
            'checkpoint', './data/SqueezeSeg/model.ckpt-23000',
            """Path to the model parameter file.""")
    tf.app.flags.DEFINE_string(
            'input_path', './data/samples/*',
            """Input lidar scan to be detected. Can process glob input such as """
            """./data/samples/*.npy or single input.""")
    tf.app.flags.DEFINE_string(
            'out_dir', './data/samples_out/', """Directory to dump output.""")



def _normalize(x):
  return (x - x.min())/(x.max() - x.min())



def detect():
  """Detect LiDAR data."""

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  with tf.Graph().as_default():
    mc = kitti_squeezeSeg_config()

    # ed: SqueezeNet의 pretrained model weight를 불러올 필요가 없고
    #     바로 SqueezeNet Model Checkpoint를 불러오면 되므로 False로 선언
    mc.LOAD_PRETRAINED_MODEL = False


    mc.BATCH_SIZE = 1  # TODO(bichen): fix this hard-coded batch size.

    model = SqueezeSeg(mc)

    # ed: 모델 가중치를 불러오는 saver 변수 선언
    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      # ed: 저장된 모델 가중치를 불러옵니다
      saver.restore(sess, FLAGS.checkpoint)

      for f in glob.iglob(FLAGS.input_path):
        lidar = np.load(f).astype(np.float32, copy=False)[:, :, :5]

        lidar_mask = np.reshape(
            (lidar[:, :, 4] > 0),
            [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
        )

        lidar = (lidar - mc.INPUT_MEAN)/mc.INPUT_STD

        pred_cls = sess.run(
            model.pred_cls,
            feed_dict={
                model.lidar_input:[lidar],
                model.keep_prob: 1.0,
                model.lidar_mask:[lidar_mask]
            }
        )

        # save the data
        file_name = f.strip('.npy').split('/')[-1]

        if IS_WINDOWS:
            np.save(
                os.path.join(FLAGS.out_dir, 'pred_windows.npy'), 
                pred_cls[0]
            )
        else:
            np.save(
                os.path.join(FLAGS.out_dir, 'pred_'+file_name+'.npy'), 
                pred_cls[0]
            )


        # save the plot
        # ed: intensity 값을 통해 depth_map을 변수에 저장한다
        depth_map = Image.fromarray(
            (255 * _normalize(lidar[:, :, 3])).astype(np.uint8))

        # ed: 예측된 레이블 포인트들의 이미지 데이터를 label_map 데이터에 저장한다
        label_map = Image.fromarray(
            (255 * visualize_seg(pred_cls, mc)[0]).astype(np.uint8))

        # ed: 위 두 이미지를 합성한다
        blend_map = Image.blend(
            depth_map.convert('RGBA'),
            label_map.convert('RGBA'),
            alpha=0.4
        )

        # ed: 최종적으로 합성한 이미지를 png 파일로 저장한다
        if IS_WINDOWS:
            blend_map.save(
                os.path.join(FLAGS.out_dir, 'plot_windows.png'))
        else:
            blend_map.save(
                os.path.join(FLAGS.out_dir, 'plot_'+file_name+'.png'))
                


def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)

  detect()

  print('Detection output written to {}'.format(FLAGS.out_dir))



if __name__ == '__main__':
    tf.app.run()
