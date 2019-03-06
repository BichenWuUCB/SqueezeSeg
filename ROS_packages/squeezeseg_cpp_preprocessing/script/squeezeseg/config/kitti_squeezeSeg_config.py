# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016
#-*- coding: utf-8 -*-

"""Model configuration for pascal dataset"""

import numpy as np
from .config import base_model_config

def kitti_squeezeSeg_config():
  """Specify the parameters to tune below."""
  mc                    = base_model_config('KITTI')

  mc.CLASSES            = ['unknown', 'car', 'pedestrian', 'cyclist']
  mc.NUM_CLASS          = len(mc.CLASSES)
  mc.CLS_2_ID           = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
  mc.CLS_LOSS_WEIGHT    = np.array([1/15.0, 1.0,  10.0, 10.0])
  mc.CLS_COLOR_MAP      = np.array([[ 0.00,  0.00,  0.00],
                                    [ 0.12,  0.56,  0.37],
                                    [ 0.66,  0.55,  0.71],
                                    [ 0.58,  0.72,  0.88]])

  mc.BATCH_SIZE         = 32


  mc.AZIMUTH_LEVEL      = 512      # for Sphrerical Projection
  mc.ZENITH_LEVEL       = 64       # for Sphrerical Projection


  mc.LCN_HEIGHT         = 3                                 # for Bilateral filter + R-CRF  (필터 사이즈)
  mc.LCN_WIDTH          = 5                                 # for Bilateral filter + R-CRF  (필터 사이즈)
  mc.BILATERAL_THETA_A  = np.array([.9, .9, .6, .6])        # for Bilateral filter  (클래스의 갯수만큼 array)
  mc.BILATERAL_THETA_R  = np.array([.015, .015, .01, .01])  # for Bilateral filter  (클래스의 갯수만큼 array)
  mc.BI_FILTER_COEF     = 0.1                               # for Bilatreal filter


  mc.RCRF_ITER          = 3                                 # for R-CRF (meanfield inference 반복 횟수)
  mc.ANG_THETA_A        = np.array([.9, .9, .6, .6])        # for R-CRF
  mc.ANG_FILTER_COEF    = 0.02                              # for R-CRF


  mc.LEARNING_RATE      = 0.01
  mc.CLS_LOSS_COEF      = 15.0           # for Loss funtion
  mc.WEIGHT_DECAY       = 0.0001         # for lr을 주기적으로 변경하는 파라미터
  mc.MAX_GRAD_NORM      = 1.0            # for gradient 수동 업데이트를 위한 파라미터
  mc.MOMENTUM           = 0.9            # for momentum Optimizer에 사용하는 파라미터
  mc.DECAY_STEPS        = 10000          # for learning_rate.exponential_decay() 함수에 사용하는 파라미터 (10000번 학습마다 lr을 낮춘다)
  mc.LR_DECAY_FACTOR    = 0.5            # for learning_rate.exponential_decay() 함수에 사용하는 파라미터


  mc.DATA_AUGMENTATION  = True    # for imdb.py data y flipping  (학습데이터 양을 늘리기 위한 기법 중 하나)
  mc.RANDOM_FLIPPING    = True    # for imdb.py data y flipping


  # x, y, z, intensity, distance
  mc.INPUT_MEAN         = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
  mc.INPUT_STD          = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])

  return mc
