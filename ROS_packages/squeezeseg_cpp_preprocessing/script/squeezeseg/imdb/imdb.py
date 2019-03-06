# Author: Bichen Wu (bichen@berkeley.edu) 02/27/2017
#-*- coding: utf-8 -*-

"""The data base wrapper class"""
import os
import random
import shutil

import numpy as np

from utils.util import *

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

  # ed: Input (.npy) 파일들을 SqueezeSeg에 맞게 불러오는 함수인듯
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

      # ed: .npy 데이터형식이 pickle보다 30배이상 빠르기 때문에 사용한다고 한다
      # load data
      # loading from npy is 30x faster than loading from pickle
      record = np.load(self._lidar_2d_path_at(idx)).astype(np.float32, copy=False)

      # ed: data augmentation을 위한 코드 (데이터의 양을 늘리기 위한 기법 중 하나)
      if mc.DATA_AUGMENTATION:
        if mc.RANDOM_FLIPPING:
          if np.random.rand() > 0.5:
            # ed: y축 순서를 좌표를 반전시키는 코드
            # flip y
            record = record[:, ::-1, :]

            # ed: y축 부호 또한 (-)로 반전시킨다
            record[:, :, 1] *= -1


      lidar = record[:, :, :5] # x, y, z, intensity, depth

      lidar_mask = np.reshape(
          (lidar[:, :, 4] > 0),
          [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
      )


      # ed: INPUT_MEAN, INPUT_STD 모두 하드코딩된 값이다. 이 값들로 lidar 데이터를 정규화시킨다
      # mc.INPUT_MEAN = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
      # mc.INPUT_STD  = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])
      # normalize
      lidar = (lidar - mc.INPUT_MEAN)/mc.INPUT_STD

      label = record[:, :, 5]
      weight = np.zeros(label.shape)


      # ed: weight 변수에 pedestrian, cyclist의 가중치를 더 크게 설정하는 코드
      #     mc.CLASSES = ['unknown', 'car', 'pedestrian', 'cyclist']
      #     mc.NUM_CLASS = 4
      #     mc.CLC_LOSS_WEIGHT = np.array([1/15.0, 1.0, 10.0, 10.0])
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
