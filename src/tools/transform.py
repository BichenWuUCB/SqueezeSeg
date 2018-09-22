#!/usr/bin/python3
# --encoding=utf8--

import component as ct
import numpy as np

import os
import time

def transform_npy(rootpath=""):
    
    print rootpath
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    
    trantool = ct.TransformData()
    trantool.rootPath = rootpath
    
    ptsfiles = trantool.load_file_names()
    
    idx = 0
    for file in ptsfiles:
        # print '正在转换 file ：%s  ......' % file
        idx += 1
        if file[-4:] == '.csv':
            prename = 'channelVELO_TOP_0000_%05d' % idx
            npyname = (prename + '.npy')
            
            npypath = trantool.savePath + npyname
            if os.path.exists(npypath):
                continue
            
            # start = time.time()
            data = trantool.cover_csv_to_nzero(file)
            formatdata = trantool.generate_image_np(data, debug=False)
            np.save(npypath, formatdata)
            
            # if np.shape(formatdata) == (64, 512, 6):
            #     print '%s 已生成' % npypath
            #     print '耗时：%s ' % time.time() - start
            
        
if __name__ == '__main__':

    path = "/home/mengweiliang/disk15/df314/training"
    transform_npy(path)
    
    