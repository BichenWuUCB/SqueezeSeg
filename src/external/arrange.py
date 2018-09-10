import os
import csv
import math
import numpy as np
from scipy.misc import imsave

def load_points(filename):
    # The data stored as csv file
    with open(filename) as f:
        reader = csv.reader(f)
        points = list(reader)
        points = np.array(points)
        return points.astype(np.float32)


def load_mask(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        mask = list(reader)
        mask = np.array(mask)
        return mask.astype(np.uint8).reshape(-1)


def save_label(filename, labels):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(labels.reshape(-1, 2))


def get_degree(x, y):
    d = math.atan2(y, x)
    d = d / math.pi * 180 + 180
    return d


def get_color(category):
    if category == 0:
        return 255, 255, 255
    elif category == 1:
        return 255, 0, 0
    elif category == 2:
        return 0, 255, 0
    elif category == 3:
        return 0, 0, 255
    elif category == 4:
        return 255, 255, 0
    elif category == 5:
        return 255, 0, 255
    elif category == 6:
        return 0, 255, 255
    else:
        return 255, 128, 0


def generate_image(points, mask, color=False, statistics=False):
    if color:
        image = np.zeros((128 * 2, 4000, 3), dtype=np.uint8)
    else:
        image = np.zeros((128, 4000), dtype=np.uint8)

    labels = np.zeros((128, 4000, 2), dtype=np.uint32)
    maxi = 0
    mini = 255
    for index, point in enumerate(points):
        x, y, z = point
        if abs(x) < 0.5 or abs(y) < 0.5:
            continue
        
        intensity = math.sqrt(x * x + y * y)

        if intensity > maxi:
            maxi = intensity
        if intensity < mini:
            mini = intensity
        d = get_degree(x, y)
        d = int(d / 0.09)
        zd = 196 - get_degree(intensity, z)
        zd = int(zd * 4)
        if zd > 127:
            zd = 127

        labels[zd][d][0] = labels[zd][d][0] * 10 + mask[index]
        labels[zd][d][1] += 1
        if color:
            r, g, b = get_color(mask[index])
            image[zd][d][0] = 255 - int(intensity)
            image[zd][d][1] = 255 - int(intensity)
            image[zd][d][2] = 255 - int(intensity)
            image[zd + 128][d][0] = r
            image[zd + 128][d][1] = g
            image[zd + 128][d][2] = b
        else:
            image[zd][d] = 255 - int(intensity)

    image = image.astype(np.uint8)

    if statistics:
        counts = labels[:, :, 1]
        print('all {}, valid {}'.format(128 * 4000 ,np.sum(counts)))
        print('max: {}'.format(np.max(counts)))
        print('point count: {}'.format(np.sum(counts > 0)))
        for i in range(np.max(counts) + 1):
            print('count {}: {}'.format(i, np.sum(counts == i)))

    return image, labels


def get_type(labels, count):
    ls = np.zeros(count, dtype=np.uint8)
    for i in range(count):
        ls[i] = labels % 10
        labels = labels / 10
    types = set(ls)
    if len(types) < 2:
        return 0
    elif len(types) == 2 and 0 in types:
        return 1
    else:
        return 2


ROOT_FOLDER = '/home/terence/repo/data/df/314/training'
SAVED_FOLDER = '../data'
filenames = next(os.walk('{}/pts/'.format(ROOT_FOLDER)))[2]   

counts = np.zeros(3, dtype=np.uint64)
maximum = np.zeros(10, dtype=np.uint32)
for name in filenames:
    points = load_points('{}/pts/{}'.format(ROOT_FOLDER, name))
    mask = load_mask('{}/category/{}'.format(ROOT_FOLDER, name))
    print('processing {}'.format(name))
    image, labels = generate_image(points, mask, statistics=True, color=True)
    imsave('{}/images/{}.png'.format(SAVED_FOLDER, name[:-4]), image)
    m = np.max(labels[:, :, 1])
    if m > 9:
        m = 9

    idx = labels[:, :, 1] > 1
    labels = labels[idx]
    count = np.zeros(3, dtype=np.uint32)
    for label in labels:
        i = get_type(label[0], label[1])
        count[i] += 1
    counts = counts + count
    maximum[m] = maximum[m] + 1
    save_label('{}/labels/{}'.format(SAVED_FOLDER, name), labels)
    print('')

print('{} {} {}'.format(counts[0], counts[1], counts[2]))
print('The max overlapping point:')
for i, count in enumerate(maximum):
    print('{}:  {}'.format(i, count))

print('Done!')
