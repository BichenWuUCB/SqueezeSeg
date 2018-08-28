# Author: Bichen Wu (bichen@berkeley.edu) 02/27/2017
#-*- coding: utf-8 -*-

"""The data base wrapper class"""
import os
import random
import shutil

import numpy as np

from squeezeseg.utils.util import *

class imdb(object):
  """Image database."""

  def __init__(self, name, mc):
    self._name = name
    self._image_set = []
    self._image_idx = []
    self._data_root_path = []
    self.mc = mc

    # batch reader
    self._perm_idx = []
    self._cur_idx = 0

  @property
  def name(self):
    return self._name

  @property
  def image_idx(self):
    return self._image_idx

  @property
  def image_set(self):
    return self._image_set

  @property
  def data_root_path(self):
    return self._data_root_path

  def _shuffle_image_idx(self):
    self._perm_idx = [self._image_idx[i] for i in
        np.random.permutation(np.arange(len(self._image_idx)))]
    self._cur_idx = 0

  def read_batch(self, shuffle=True):
    """Read a batch of lidar data including labels. Data formated as numpy array
    of shape: height x width x {x, y, z, intensity, range, label}.

    Args:
      shuffle: whether or not to shuffle the dataset

    Returns:
      lidar_per_batch: LiDAR input. Shape: batch x height x width x 5.
      lidar_mask_per_batch: LiDAR mask, 0 for missing data and 1 otherwise.
        Shape: batch x height x width x 1.
      label_per_batch: point-wise labels. Shape: batch x height x width.
      weight_per_batch: loss weights for different classes. Shape: 
        batch x height x width
    """
    mc = self.mc

    if shuffle:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        self._shuffle_image_idx()
      batch_idx = self._perm_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
      self._cur_idx += mc.BATCH_SIZE

    else:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        batch_idx = self._image_idx[self._cur_idx:] \
            + self._image_idx[:self._cur_idx + mc.BATCH_SIZE-len(self._image_idx)]
        self._cur_idx += mc.BATCH_SIZE - len(self._image_idx)

      else:
        batch_idx = self._image_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
        self._cur_idx += mc.BATCH_SIZE


    lidar_per_batch = []
    lidar_mask_per_batch = []
    label_per_batch = []
    weight_per_batch = []

    for idx in batch_idx:

      # load data
      # loading from npy is 30x faster than loading from pickle
      record = np.load(self._lidar_2d_path_at(idx)).astype(np.float32, copy=False)

      if mc.DATA_AUGMENTATION:
        if mc.RANDOM_FLIPPING:
          if np.random.rand() > 0.5:
            # flip y
            record = record[:, ::-1, :]

            record[:, :, 1] *= -1


      lidar = record[:, :, :5] # x, y, z, intensity, depth

      lidar_mask = np.reshape(
          (lidar[:, :, 4] > 0),
          [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
      )


      # normalize
      lidar = (lidar - mc.INPUT_MEAN)/mc.INPUT_STD

      label = record[:, :, 5]
      weight = np.zeros(label.shape)


      for l in range(mc.NUM_CLASS):
        weight[label==l] = mc.CLS_LOSS_WEIGHT[int(l)]


      # Append all the data
      lidar_per_batch.append(lidar)
      lidar_mask_per_batch.append(lidar_mask)
      label_per_batch.append(label)
      weight_per_batch.append(weight)


    return np.array(lidar_per_batch), np.array(lidar_mask_per_batch), \
        np.array(label_per_batch), np.array(weight_per_batch)

  def evaluate_detections(self):
    raise NotImplementedError
