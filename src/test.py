#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:18-9-28 下午8:16
# software:PyCharm

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

import pandas as pd

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint', '../scripts/log/train/model.ckpt-75000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', '../data/test/npy/*',
    """Input lidar scan to be detected. Can process glob input such as """
    """./data/samples/*.npy or single input.""")

tf.app.flags.DEFINE_string(
    'out_dir', '../scripts/log/answers/', """Directory to dump output.""")
tf.app.flags.DEFINE_string('gpu', '4', """gpu id.""")


# my code
def geneate_results():
    
    pass


def test():
    """Detect LiDAR data."""
    
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    
    with tf.Graph().as_default():
        mc = alibaba_squeezeSeg_config()
        mc.LOAD_PRETRAINED_MODEL = False
        mc.BATCH_SIZE = 1  # TODO(bichen): fix this hard-coded batch size.
        model = SqueezeSeg(mc)
        
        saver = tf.train.Saver(model.model_params)
        
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver.restore(sess, FLAGS.checkpoint)
            
            def generate_pred_cls(f, mc, model, sess):
                
                lidar = f
    
                lidar_mask = np.reshape(
                    (lidar[:, :, 4] > 0),
                    [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
                )
                lidar = (lidar - mc.INPUT_MEAN) / mc.INPUT_STD
    
                pred_cls = sess.run(
                    model.pred_cls,
                    feed_dict={
                        model.lidar_input: [lidar],
                        model.keep_prob: 1.0,
                        model.lidar_mask: [lidar_mask]
                    }
                )
                
                return pred_cls
            
            
            for f in glob.iglob(FLAGS.input_path):
                # save the data
                file_name = f.strip('.npy').split('/')[-1]
                file_path = FLAGS.out_dir + file_name + '.csv'
                
                if os.path.exists(file_path):
                    print(file_path)
                    continue
                
                fnpy = np.load(f).astype(np.float32, copy=False)
                
                if np.shape(fnpy)[0] >= 32768:
                    
                    f1 = np.load(f).astype(np.float32, copy=False)[:32768, :5]
                    f1 = np.reshape(f1, (64, 512, 5))
                    
                    fillnp = np.zeros((32768, 5)).astype(np.float32)
                    f2 = np.load(f).astype(np.float32, copy=False)[32768:, :5]
                    avildable_number = np.shape(f2)[0]
                    padding_number = 32768 - avildable_number   # adding number
                    fillnp[:np.shape(f2)[0], :5] = f2[:]
                    
                    # print(np.shape(fnpy))
                    # print(np.shape(f1), np.shape(fillnp))
                    
                    fillnp = np.reshape(fillnp, (64, 512, 5))
                    
                    pred_cls1 = generate_pred_cls(f1, mc, model, sess)
                    pred_cls2 = generate_pred_cls(fillnp, mc, model, sess)
                
                    result1 = np.reshape(pred_cls1, (32768, 1))
                    result2 = np.reshape(pred_cls2, (32768, 1))
                
                    result = np.zeros((np.shape(fnpy)[0], 1)).astype(np.float32, copy=True)
                    result[:32768, :] = result1
                    result[32768:(32768+avildable_number), :] = result2[:avildable_number, :]
                    
                else:
                    
                    f1 = np.zeros((32768, 5))
                    avildable_number = np.shape(fnpy)[0]
                    f1[:np.shape(fnpy)[0], :5] = fnpy[:, :5]
                    
                    f1 = np.reshape(f1, (64, 512, 5))
                    pred_cls = generate_pred_cls(f1, mc, model, sess)
                    
                    result = np.reshape(pred_cls, (32768, 1))
                    result = result[:avildable_number, :]
                    
                
                # print(file_name)
                # print(pred_cls)

                pdata = pd.DataFrame(np.reshape(result, (-1, 1)),columns=['category'])
                
                if not os.path.exists(file_path):
                    pdata[['category']].astype('int32').to_csv(file_path, index=None, header=None)
                # np.save(
                #     os.path.join(FLAGS.out_dir, 'pred_' + file_name + '.npy'),
                #     pred_cls[0]
                # )
                
                

def main(argv=None):
    if not tf.gfile.Exists(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)
    
    print('Detection output written to {}'.format(FLAGS.out_dir))
    test()

if __name__ == '__main__':
    tf.app.run()
