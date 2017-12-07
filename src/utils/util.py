# Author: Bichen Wu (bichen@berkeley.edu) 02/20/2017

"""Utility functions."""

import numpy as np
import time
from matplotlib import cm
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import DBSCAN

def bgr_to_rgb(ims):
  """Convert a list of images from BGR format to RGB format."""
  out = []
  for im in ims:
    out.append(im[:,:,::-1])
  return out

class Timer(object):
  def __init__(self):
    self.total_time   = 0.0
    self.calls        = 0
    self.start_time   = 0.0
    self.duration     = 0.0
    self.average_time = 0.0

  def tic(self):
    self.start_time = time.time()

  def toc(self, average=True):
    self.duration = time.time() - self.start_time
    self.total_time += self.duration
    self.calls += 1
    self.average_time = self.total_time/self.calls
    if average:
      return self.average_time
    else:
      return self.duration

def conf_error_rate_at_thresh_fn(mask, conf, thresh):
  return np.mean((conf>thresh) != mask)

def auc_roc_fn(mask, conf):
  label = mask.reshape((-1))
  score = conf.reshape((-1))
  fpr, tpr, _ = roc_curve(label, score)
  return auc(fpr, tpr)

def rmse_fn(diff, nnz):
  return np.sqrt(np.sum(diff**2)/nnz)

def abs_accuracy_at_thresh_fn(diff, thresh, mask):
  return np.sum((np.abs(diff) < thresh)*mask)/float(np.sum(mask))

def rel_accuracy_at_thresh_fn(pred_ogm, gt_ogm, mask, thresh):
  return np.sum(
      mask * (np.maximum(pred_ogm, gt_ogm) / 
              np.minimum(gt_ogm, pred_ogm) < thresh)
      )/float(np.sum(mask))
  

def mat_to_img(mat):
  # - convert to numpy array
  # - convert from grey-scale to color map with format RGBA
  # - Toggle along X-axis so it can be shown properly on tensorboard
  return cm.viridis(np.array(mat, dtype=np.float))[:, ::-1, :, :]

def evaluate_iou(label, pred, n_class, epsilon=1e-12):
  """Evaluation script to compute pixel level IoU.

  Args:
    label: N-d array of shape [batch, W, H], where each element is a class
        index.
    pred: N-d array of shape [batch, W, H], the each element is the predicted
        class index.
    n_class: number of classes
    epsilon: a small value to prevent division by 0

  Returns:
    IoU: array of lengh n_class, where each element is the average IoU for this
        class.
    tps: same shape as IoU, where each element is the number of TP for each
        class.
    fps: same shape as IoU, where each element is the number of FP for each
        class.
    fns: same shape as IoU, where each element is the number of FN for each
        class.
  """

  assert label.shape == pred.shape, \
      'label and pred shape mismatch: {} vs {}'.format(
          label.shape, pred.shape)

  ious = np.zeros(n_class)
  tps = np.zeros(n_class)
  fns = np.zeros(n_class)
  fps = np.zeros(n_class)

  for cls_id in range(n_class):
    tp = np.sum(pred[label == cls_id] == cls_id)
    fp = np.sum(label[pred == cls_id] != cls_id)
    fn = np.sum(pred[label == cls_id] != cls_id)

    ious[cls_id] = tp/(tp+fn+fp+epsilon)
    tps[cls_id] = tp
    fps[cls_id] = fp
    fns[cls_id] = fn

  return ious, tps, fps, fns

def evaluate_instance_iou(
    gt_cluster_per_batch, pr_cluster_per_batch, n_class, iou_thresh=0.3,
    epsilon=1e-12):
  """Evaluate instance IOU.

  Args:
    gt_cluster_per_batch: ground truth labels. A nested list of size: batch x
        class x obj x pts.
    pr_cluster_per_batch: predicted labels. Same as above.
    n_class: number of classes
    epsilon: a small value to prevent division by 0

  Returns:
    IoU: array of n_class, where each element is the average IoU for this class.
    tps: same shape as IoU, where each element is the number of TP for each
        class.
    fps: same shape as IoU, where each element is the number of FP for each`
        class.
    fns: same shape as IoU, where each element is the number of FN for each
        class.
  """
  tps = np.zeros(n_class)
  fns = np.zeros(n_class)
  fps = np.zeros(n_class)

  otps = np.zeros(n_class)
  ofns = np.zeros(n_class)
  ofps = np.zeros(n_class)


  for cls_id in range(n_class):
    for b in range(len(gt_cluster_per_batch)):
      iou, tp, fp, fn, otp, ofp, ofn = compute_instance_iou(
          gt_cluster_per_batch[b][cls_id],
          pr_cluster_per_batch[b][cls_id],
          iou_thresh
      )
      tps[cls_id] += tp
      fps[cls_id] += fp 
      fns[cls_id] += fn 

      otps[cls_id] += otp
      ofps[cls_id] += ofp 
      ofns[cls_id] += ofn 

    ious = tps/(tps+fps+fns+epsilon)
  return ious, tps, fps, fns, otps, ofps, ofns

def compute_instance_iou(
    gtInstanceList, prInstanceList, iou_thresh, epsilon=1e-12,
    min_pts_thresh=0):
  """Compute instance level IOU.

  Args:
    gtInstanceList: a list of set of points representing ground truth instance
      labels. The format is the following: 
        [instance_0, instance_1, ..., instance_N]
      where instance_i is list of pt coordinates {pt_1, pt_2, ... pt_M}. pt_j is
      the index to a point that belongs to a instance. The points in each
      instance should be mutually exclusive.
    prInstanceList: Same as above, but contains predicted labels.
    iou_thresh: an IOU threshold to determine if an object is detected
    epsilon: a small value to prevent division by 0

  Returns:
    ins_iou: Weighted average of IOU score for instance-wise segmentation
    tps: number of true positive points
    fps: number of false positive points
    fns: number of false negative points
    otps: number of true positive instances
    ofps: number of false positive instances
    ofns: number of false negative instances
  """
  # assure that points between instances are mutually exclusive
  def check_mutually_exclusive(instanceList):
    is_mutually_exclusive = True
    for i in range(len(instanceList)):
      for j in range(i+1, len(instanceList)):
        if len(instanceList[i].intersection(instanceList[j])) > 0:
          is_mutually_exclusive = False
          break
      if not is_mutually_exclusive:
        break
    assert is_mutually_exclusive, \
        'Instances are not mutually exclusive'

  # check_mutually_exclusive(gtInstanceList)
  # check_mutually_exclusive(prInstanceList)

  # Rank gt instances by their cardinality (#pts in each instance)
  gtInstanceList.sort(key=lambda x: len(x), reverse=True)

  # Match each gt instance with a predicted instance with the largest IOU with
  # it. Then, compute the IOU as well as the union of this match. The
  # output is just the weighted average of the iou for each match.
  matched = [False]*len(prInstanceList)
  tps, fps, fns = 0., 0., 0.
  otps, ofps, ofns = 0., 0., 0.
  for gidx, gint in enumerate(gtInstanceList):
    max_iou = 0.
    matched_tp, matched_fp, matched_fn = 0., 0., 0.
    matched_idx = -1
    for pidx, pint in enumerate(prInstanceList):
      if matched[pidx]:
        continue
      tp = len(gint.intersection(pint))
      fp = len(pint.difference(gint))
      # assert fp+tp == len(pint), 'fp+tp != #pr'
      fn = len(gint.difference(pint))
      union = len(gint.union(pint))
      # assert union == tp+fp+fn, 'union != tp+fp+fn'
      iou = tp/(tp+fp+fn+epsilon)
      if iou > max_iou:
        max_iou = iou
        matched_idx = pidx
        matched_tp, matched_fp, matched_fn = tp, fp, fn
    if matched_idx != -1:
      # only count gt if it's larger than certain threshold
      # if len(gint) > min_pts_thresh:
      matched[matched_idx] = True
      tps += matched_tp
      fps += matched_fp
      fns += matched_fn
      if max_iou > iou_thresh:
        otps += 1
      else:
        ofps += 1
    else:
      fns += len(gint)
      ofns += 1

  
  # deal with un-matched false positive instances
  for pidx, pint in enumerate(prInstanceList):
    if not matched[pidx]:
      fps += len(pint)
      ofps += 1

  ins_iou = tps/(tps+fps+fns+epsilon)

  return ins_iou, tps, fps, fns, otps, ofps, ofns

def cluster_gt_point_cloud(
    pts, labels, cls_ids, min_pts_thresh=0, max_dist_thresh=100):
  assert pts.shape[:2] == labels.shape, 'label and pts shape mismatch'
  cluster_per_scan = []
  count = 0
  instance_id = {}
  for cid in cls_ids:
    X = pts[labels == cid, :]
    indices = np.argwhere((labels == cid))
    cluster_ids = []
    for pt in X:
      tpt = tuple(pt)
      if tpt not in instance_id:
        count += 1
        instance_id[tpt] = count
      cluster_ids.append(instance_id[tpt])
    cluster_ids = np.array(cluster_ids)
    cluster_per_class = [] 
    for l in set(cluster_ids):
      instance = indices[cluster_ids==l]
      dist = np.linalg.norm(pts[instance[0][0], instance[0][1]][:2])
      if len(instance) < min_pts_thresh or dist > max_dist_thresh:
        for index in instance:
          labels[index[0], index[1]] = 0
        continue
      instance = set(map(tuple, instance))
      cluster_per_class.append(instance)
    cluster_per_scan.append(cluster_per_class)

  return cluster_per_scan

def cluster_point_cloud(
    pts, labels, cls_ids, radius=0.3, min_pts=3, min_cluster_pts=20):
  """Cluster point cloud into instances.

  Args:
    pts: N-d array of shape [ZENITH, AZIMUTH, d]. It contains point coordinates.
    labels: N-d array of shape [ZENITH, AZIMUTH] where each element is the label
        of the point.
    cls_ids: array of class ids to perform clustering on.
    radius: parameter for DBSCAN. 
    min_pts: parameter for DBSCAN.
    min_cluster_pts: minimum number of points in a cluster

  Returns:
    bbox: array of num_cls x num_cluster x 4, where 4 are bounding box
        parameters [xmin, ymin, xmax, ymax] for each cluster
    cluster: array of num_cls * num_cluster * num_pts, where the last dimension
        is set of points.
  """

  assert pts.shape[:2] == labels.shape, 'label and pts shape mismatch'

  if not isinstance(radius, (list, tuple, dict)):
    radius = dict(zip(cls_ids, [radius]*len(cls_ids)))
  if not isinstance(min_cluster_pts, (list, tuple, dict)):
    min_cluster_pts = dict(zip(cls_ids, [min_cluster_pts]*len(cls_ids)))

  bbox_per_scan = []
  cluster_per_scan = []
  for cid in cls_ids:
    bbox_per_class = []
    cluster_per_class = [] 
    X = pts[labels == cid, :]
    indices = np.argwhere((labels == cid))

    # num_orig_pts = len(indices)
    # num_ins_pts = 0
    # num_filtered_pts = 0

    if X.shape[0] < min_cluster_pts[cid]:
      # filter noisy points
      for ij in indices:
        labels[ij[0], ij[1]] = 0
    else:
      db = DBSCAN(eps=radius[cid], min_samples=min_pts).fit(X)
      cluster_ids = db.labels_
      cluster_id_set = set(cluster_ids)

      for l in cluster_id_set:
        # set far points' label to 0

        if l == -1 or np.sum(cluster_ids==l) < min_cluster_pts[cid]:
          # filter noisy points
          for ij in indices[cluster_ids==l]:
            labels[ij[0], ij[1]] = 0
          continue

        # convert point indices into a set
        instance = indices[cluster_ids==l]
        instance = set(map(tuple, instance))
        cluster_per_class.append(instance)
        # num_ins_pts += len(instance)

        # comptue bbox
        xmin = np.min(indices[cluster_ids==l, 0])
        xmax = np.max(indices[cluster_ids==l, 0])
        ymin = np.min(indices[cluster_ids==l, 1])
        ymax = np.max(indices[cluster_ids==l, 1])
        bbox_per_class.append([xmin, ymin, xmax, ymax])
      
    bbox_per_scan.append(bbox_per_class)
    cluster_per_scan.append(cluster_per_class)

  return bbox_per_scan, cluster_per_scan

def condensing_matrix(size_z, size_a, in_channel):
  assert size_z % 2 == 1 and size_a % 2==1, \
      'size_z and size_a should be odd number'

  half_filter_dim = (size_z*size_a)//2

  # moving neigboring pixels to channel dimension
  nbr2ch_mat = np.zeros(
      (size_z, size_a, in_channel, size_z*size_a*in_channel),
      dtype=np.float32
  )
  
  for z in range(size_z):
    for a in range(size_a):
      for ch in range(in_channel):
        nbr2ch_mat[z, a, ch, z*(size_a*in_channel) + a*in_channel + ch] = 1
  
  # exclude the channel index corresponding to the center position
  nbr2ch_mat = np.concatenate(
      [nbr2ch_mat[:, :, :, :in_channel*half_filter_dim], 
       nbr2ch_mat[:, :, :, in_channel*(half_filter_dim+1):]],
      axis=3
  )
   
  assert nbr2ch_mat.shape == \
      (size_z, size_a, in_channel, (size_a*size_z-1)*in_channel), \
      'error with the shape of nbr2ch_mat after removing center position'

  return nbr2ch_mat

def angular_filter_kernel(size_z, size_a, in_channel, theta_sqs):
  """Compute a gaussian kernel.
  Args:
    size_z: size on the z dimension.
    size_a: size on the a dimension.
    in_channel: input (and output) channel size
    theta_sqs: an array with length == in_channel. Contains variance for
        gaussian kernel for each channel.
  Returns:
    kernel: ND array of size [size_z, size_a, in_channel, in_channel], which is
        just guassian kernel parameters for each channel.
  """
  assert size_z % 2 == 1 and size_a % 2==1, \
      'size_z and size_a should be odd number'
  assert len(theta_sqs) == in_channel, \
      'length of theta_sqs and in_channel does no match'

  # gaussian kernel
  kernel = np.zeros((size_z, size_a, in_channel, in_channel), dtype=np.float32)
  for k in range(in_channel):
    kernel_2d = np.zeros((size_z, size_a), dtype=np.float32)
    for i in range(size_z):
      for j in range(size_a):
        diff = np.sum(
            (np.array([i-size_z//2, j-size_a//2]))**2)
        kernel_2d[i, j] = np.exp(-diff/2/theta_sqs[k])

    # exclude the center position
    kernel_2d[size_z//2, size_a//2] = 0
    kernel[:, :, k, k] = kernel_2d

  return kernel
