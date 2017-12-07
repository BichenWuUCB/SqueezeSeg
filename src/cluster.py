import os
from PIL import Image, ImageDraw

from utils.util import *

color_map = np.array(
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


data_root = '/rscratch/bichen/data/KITTI_raw/2011_09_26/2011_09_26_drive_0005_sync/seg/data'
cluster_root = '/rscratch/bichen/data/KITTI_raw/2011_09_26/2011_09_26_drive_0005_sync/seg/cluster_img'

# os.makedirs(cluster_root)

timer = Timer()

for scanfile in os.listdir(data_root):
  if scanfile[-3:] == 'npy':
    frame = np.load(os.path.join(data_root, scanfile))
    # pts = frame[:, :, :3]
    pts = frame[:, :, 6:8]
    labels = frame[:, :, 5]

    timer.tic()
    bbox_per_scan, clusters = cluster_point_cloud(
        pts, labels,
        [1, 2, 4, 6], # car, van, pedestrian, cyclist
    )
    timer.toc()
    print "Average time for clustering {}".format(timer.average_time)

    # plot segmentation
    class_mask = np.zeros((frame.shape[0], frame.shape[1], 3))
    # for i in range(class_mask.shape[0]):
    #   for j in range(class_mask.shape[1]):
    #     class_mask[i, j, :] = color_map[int(labels[i, j])]
    # plot instance segmentation
    visited = set()
    for c in clusters:
      for i in c:
        color = np.random.rand(3)
        for x, y in i:
          if (x, y) in visited:
            print (x, y)
          
          visited.add((x, y))
          class_mask[x, y, :] = color

    im = Image.fromarray((255.0 * class_mask).astype(np.uint8))

    draw = ImageDraw.Draw(im)

    # plot clustering bbox
    for bbox_per_cls in bbox_per_scan:
      for bbox in bbox_per_cls:
        draw.rectangle(((bbox[1], bbox[0]), (bbox[3], bbox[2])), outline='red')

    im.save(os.path.join(
      cluster_root, scanfile[:-3]+'png'))
