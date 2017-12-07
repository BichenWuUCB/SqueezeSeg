# Author: Bichen Wu (bichen@berkeley.edu) 02/27/2017

"""The data base wrapper class"""

import os
import random
import shutil

from PIL import Image, ImageFont, ImageDraw
import cv2
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

  def read_image_batch(self, shuffle=True):
    """Only Read a batch of images without occupancy grid maps
    Args:
      shuffle: whether or not to shuffle the dataset
    Returns:
      images: length batch_size list of arrays [height, width, 3]
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

    images = []
    for i in batch_idx:
      im = cv2.imread(self._image_path_at(i))
      im = im.astype(np.float32, copy=False)
      im -= mc.BGR_MEANS
      im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
      images.append(im)

    return images

  def data_augmentation(self, record):
    mc = self.mc
    xyz = record[:, :, :3]
    mask = np.reshape(
        (record[:, :, 4] > 0), 
        [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
    )

    # generate a random shift and update data
    dxyz = (np.random.rand(1, 1, 3)-0.5)
    dxyz[:, :, 0] *= mc.DELTA_X_RANGE
    dxyz[:, :, 1] *= mc.DELTA_Y_RANGE
    dxyz[:, :, 2] *= mc.DELTA_Z_RANGE

    xyz += mask * dxyz
    r = np.apply_along_axis(np.linalg.norm, 2, xyz)
    d = np.apply_along_axis(np.linalg.norm, 2, xyz[:, :, :2])

    record[:, :, :3]  = xyz
    record[:, :, 4] = r
    record[:, :, 6:9] += mask * dxyz

    # compute new target positions
    with np.errstate(divide='ignore', invalid='ignore'):
      zenith = (-np.nan_to_num(np.arcsin(xyz[:, :, 2]/r))*180/np.pi \
                + mc.ZENITH_MAX) * (mc.ZENITH_LEVEL/mc.ZENITH_RANGE)
      azimuth = (-np.nan_to_num(np.arcsin(xyz[:, :, 1]/d))*180/np.pi \
                + mc.AZIMUTH_MAX) * (mc.AZIMUTH_LEVEL/mc.AZIMUTH_RANGE)

    target_idx = np.transpose(
        np.stack((zenith, azimuth)), (1, 2, 0)).astype(np.int)
     
    # valid target positions
    new_mask = np.squeeze(mask) \
             * (xyz[:, :, 0] > 0) \
             * (zenith >= 0 ) \
             * (zenith < mc.ZENITH_LEVEL) \
             * (azimuth >=0) \
             * (azimuth < mc.AZIMUTH_LEVEL) \

    c1, c2 = target_idx[new_mask, :].T
    new_record = np.zeros(record.shape)
    new_record[c1, c2] = record[new_mask, :]

    return new_record

  def read_batch(self, shuffle=True):
    """Read a batch of images with corresponding grid occupancy maps
    Args:
      shuffle: whether or not to shuffle the dataset
    Returns:
      image_per_batch: images. Shape: batch_size x height x width x [b, g, r]
      dpm_per_batch: depth maps. Shape: batch_size x height x width
      mask_per_batch: a boolean tensor where each element represent if the data
          is present or missing. Shape: batch_size x height x width
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
    label_mask_per_batch = []
    label_per_batch = []
    weight_per_batch = []
    center_xyz_per_batch = []
    cluster_per_batch = []

    if mc.DEBUG_MODE:
      nz_dpm = []

    for idx in batch_idx:
      # load data
      # loading from npy is 30x faster than loading from pickle
      record = np.load(self._lidar_2d_path_at(idx)).astype(np.float32, copy=False)

      # TODO(bichen): fix this hard-code
      if idx[:4] != 'gta_':
        # reshape the input tensor
        record = record[:, 256:-256, :]
        assert record.shape[:2] == (mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL), \
            'Data spatial shape mismatch: {} (should be {}).'.format(
                record.shape[:2], (mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL))

      if mc.DATA_AUGMENTATION:
        if mc.RANDOM_SHIFT:
          record = self.data_augmentation(record)
        if mc.RANDOM_FLIPPING:
          if np.random.rand() > 0.5:
            record = record[:, ::-1, :]
            # flip y
            record[:, :, 1] *= -1
            # flip cy
            record[:, :, 7] *= -1

      lidar = record[:, :, :5] # x, y, z, intensity, depth
      lidar_mask = np.reshape(
          (lidar[:, :, 4] > 0), 
          [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
      )
      # normalize
      lidar = (lidar - mc.INPUT_MEAN)/mc.INPUT_STD

      # squeeze out intensity
      lidar = np.delete(lidar, 3, axis=2)

      label = record[:, :, 5]
      # convert old class idx to new class index
      for l in range(mc.ORIG_NUM_CLASS):
        label[label==l] = mc.CLS_IDX_CONVERSION[l]

      label_mask = (label > 0).reshape(
          (mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1)).astype(np.float32)

      weight = np.zeros(label.shape)
      for l in range(mc.NUM_CLASS):
        weight[label==l] = mc.CLS_LOSS_WEIGHT[int(l)]

      # center coordinates
      center_xyz = record[:, :, 6:9]

      # clusters
      # TODO(bichen): here label is also modified. Modify or document this
      # properly
      clusters = cluster_gt_point_cloud(
          center_xyz, label, range(1, mc.NUM_CLASS))

      # Append all the data
      lidar_per_batch.append(lidar)
      lidar_mask_per_batch.append(lidar_mask)
      label_per_batch.append(label)
      label_mask_per_batch.append(label_mask)
      weight_per_batch.append(weight)
      center_xyz_per_batch.append(center_xyz)
      cluster_per_batch.append(clusters)

    return np.array(lidar_per_batch), np.array(lidar_mask_per_batch), \
        np.array(label_per_batch), np.array(label_mask_per_batch), \
        np.array(weight_per_batch), np.array(center_xyz_per_batch), \
        cluster_per_batch

  def evaluate_detections(self):
    raise NotImplementedError
