# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for pascal dataset"""

import numpy as np

from config import base_model_config

def kitti_squeezeSeg_config():
  """Specify the parameters to tune below."""
  mc                    = base_model_config('KITTI')

  mc.ORIG_CLASSES       = mc.CLASSES
  mc.ORIG_NUM_CLASS     = mc.NUM_CLASS

  # mc.CLASSES            = [
  #     'unknown', 'car', 'van', 'truck', 'pedestrian', 'cyclist']
  mc.CLASSES            = ['unknown', 'car']
  mc.NUM_CLASS          = len(mc.CLASSES)
  mc.CLS_2_ID           = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
  # mc.CLS_IDX_CONVERSION = [0, 1, 2, 3, 4, 0, 5, 0, 0]
  mc.CLS_IDX_CONVERSION = [0, 1, 1, 1, 0, 0, 0, 0, 0]

  # mc.CLS_LOSS_WEIGHT    = np.array([1/15.0, 1.0,  2.0, 3.0, 6.0, 6.0])
  # mc.CLS_LOSS_WEIGHT    = np.array([1/15.0, 1.0,  10.0, 10.0])
  mc.CLS_LOSS_WEIGHT    = np.array([1/10.0, 1.0])
  # mc.CLS_COLOR_MAP      = np.array(
  #     [[ 0.00,  0.00,  0.00],
  #      [ 0.12,  0.56,  0.37],
  #      [ 0.66,  0.55,  0.71],
  #      [ 0.58,  0.72,  0.88],
  #      [ 0.25,  0.51,  0.76],
  #      [ 0.40,  0.19,  0.10]]
  # )
  mc.CLS_COLOR_MAP      = np.array(
      [[ 0.00,  0.00,  0.00],
       [ 0.12,  0.56,  0.37]]
  )

  mc.BATCH_SIZE         = 16
  mc.AZIMUTH_LEVEL      = 512
  mc.ZENITH_LEVEL       = 64
  mc.AZIMUTH_MAX        = 45.
  mc.AZIMUTH_MIN        = -45.
  mc.AZIMUTH_RANGE      = mc.AZIMUTH_MAX - mc.AZIMUTH_MIN
  mc.ZENITH_MAX         = 2.5
  mc.ZENITH_MIN         = -23.5
  mc.ZENITH_RANGE       = mc.ZENITH_MAX - mc.ZENITH_MIN

  mc.LCN_HEIGHT         = 3
  mc.LCN_WIDTH          = 5
  mc.RCRF_ITER          = 3
  # mc.BILATERAL_THETA_I  = np.array([.15, .15, .15, .15, .1, .1])
  # mc.BILATERAL_THETA_R  = np.array([.015, .015, .015, .015, .01, .01])
  mc.BILATERAL_THETA_A  = np.array([.9, .9])
  mc.BILATERAL_THETA_R  = np.array([.015, .015])
  mc.BI_FILTER_COEF     = 0.1
  # mc.ANG_THETA_A        = np.array([.9, .9, .9, .9, .9, 9])
  mc.ANG_THETA_A        = np.array([.9, .9])
  mc.ANG_FILTER_COEF    = 0.02

  # mc.DBSCAN_MIN_PTS     = [0, 20, 20, 20, 20, 20]
  # mc.DBSCAN_RADIUS      = [0, .3, .3, .3, .05, .05]
  mc.DBSCAN_MIN_PTS     = [0, 20]
  mc.DBSCAN_RADIUS      = [0, .3]

  mc.PLOT_X_RANGE       = 600
  mc.PLOT_Y_RANGE       = 800
  mc.PLOT_RESOLUTION    = 0.1

  mc.CLS_LOSS_COEF      = 15.0
  mc.CENTER_REG_COEF    = 1.0
  mc.CENTER_COS_COEF    = 5.0
  mc.WEIGHT_DECAY       = 0.0001
  mc.LEARNING_RATE      = 0.01
  mc.DECAY_STEPS        = 10000
  mc.MAX_GRAD_NORM      = 1.0
  mc.MOMENTUM           = 0.9
  mc.LR_DECAY_FACTOR    = 0.5

  mc.DATA_AUGMENTATION  = True
  mc.RANDOM_SHIFT       = False
  mc.RANDOM_FLIPPING    = True
  mc.DELTA_X_RANGE      = 5
  mc.DELTA_Y_RANGE      = 5
  mc.DELTA_Z_RANGE      = 0.5

  mc.CONF_THRESH        = 0.5
  mc.ERROR_THRESH       = 1.5
  mc.REL_ERROR_THRESH   = 1.25

  # x, y, z, intensity, distance
  mc.INPUT_MEAN         = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
  mc.INPUT_STD          = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])

  return mc
