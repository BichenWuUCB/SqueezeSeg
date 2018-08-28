#G Author: Bichen Wu (bichen@berkeley.edu) 02/20/2017
#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""Base Model configurations"""

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

def base_model_config(dataset='KITTI'):
  assert dataset.upper()=='KITTI', \
      'Currently only support KITTI dataset'

  cfg = edict()

  # Dataset used to train/val/test model. Now support KITTI
  cfg.DATASET = dataset.upper()

  # classes
  cfg.CLASSES = [
      'unknown',
      'car',
      'van',
      'truck',
      'pedestrian',
      'person_sitting',
      'cyclist',
      'tram',
      'misc',
  ]

  # number of classes
  cfg.NUM_CLASS = len(cfg.CLASSES)

  # dict from class name to id
  cfg.CLS_2_ID = dict(zip(cfg.CLASSES, range(len(cfg.CLASSES))))

  # loss weight for each class
  cfg.CLS_LOSS_WEIGHT = np.array(
      [1/20.0, 1.0,  2.0, 3.0,
       8.0, 10.0, 8.0, 2.0, 1.0]
  )

  # rgb color for each class
  cfg.CLS_COLOR_MAP = np.array(
      [[ 0.00,  0.00,  0.00],
       [ 0.12,  0.56,  0.37],
       [ 0.66,  0.55,  0.71],
       [ 0.58,  0.72,  0.88],
       [ 0.25,  0.51,  0.76],
       [ 0.98,  0.47,  0.73],
       [ 0.40,  0.19,  0.10],
       [ 0.87,  0.19,  0.17],
       [ 0.13,  0.55,  0.63]]
  )

  # Probability to keep a node in dropout
  cfg.KEEP_PROB = 0.5

  # image width
  cfg.IMAGE_WIDTH = 224

  # image height
  cfg.IMAGE_HEIGHT = 224

  # number of vertical levels
  cfg.NUM_LEVEL = 10

  # number of pie sectors of the field of view
  cfg.NUM_SECTOR = 90

  # maximum distance of a measurement
  cfg.MAX_DIST = 100.0

  # batch size
  cfg.BATCH_SIZE = 20

  # Pixel mean values (BGR order) as a (1, 1, 3) array. Below is the BGR mean
  # of VGG16
  cfg.BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])

  # Pixel mean values (BGR order) as a (1, 1, 3) array. Below is the BGR mean
  # of VGG16
  cfg.RGB_MEANS = np.array([[[123.68, 116.779, 103.939]]])

  # reduce step size after this many steps
  cfg.DECAY_STEPS = 10000

  # multiply the learning rate by this factor
  cfg.LR_DECAY_FACTOR = 0.1

  # learning rate
  cfg.LEARNING_RATE = 0.005

  # momentum
  cfg.MOMENTUM = 0.9

  # weight decay
  cfg.WEIGHT_DECAY = 0.0005

  # wether to load pre-trained model
  cfg.LOAD_PRETRAINED_MODEL = True

  # path to load the pre-trained model
  cfg.PRETRAINED_MODEL_PATH = ''

  # print log to console in debug mode
  cfg.DEBUG_MODE = False

  # gradients with norm larger than this is going to be clipped.
  cfg.MAX_GRAD_NORM = 10.0

  # Whether to do data augmentation
  cfg.DATA_AUGMENTATION = False

  # The range to randomly shift the image widht
  cfg.DRIFT_X = 0

  # The range to randomly shift the image height
  cfg.DRIFT_Y = 0

  # small value used in batch normalization to prevent dividing by 0. The
  # default value here is the same with caffe's default value.
  cfg.BATCH_NORM_EPSILON = 1e-5

  # small value used in denominator to prevent division by 0
  cfg.DENOM_EPSILON = 1e-12

  # capacity for tf.FIFOQueue
  cfg.QUEUE_CAPACITY = 80

  cfg.NUM_ENQUEUE_THREAD = 8

  return cfg
