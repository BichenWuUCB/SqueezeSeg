#!/usr/bin/python3
# --encoding=utf8--

import os
import numpy as np
import pandas as pd
import cmath, math
import time

class TransformData(object):
    
    # 数据根目录
    @property
    def rootPath(self):
        return self._rootPath
    
    @rootPath.setter
    def rootPath(self, path):
        self._rootPath = path
        
    # 生成的npy数据保存目录
    @property
    def savePath(self):
        return self.rootPath + '/npy/'

    # 转换数据 并 生成npy文件
    def transformData(self, filepath=""):
        assert filepath != ""
        
        

    def __init__(self):
        pass

    # 加载所有的文件名
    def load_file_names(self):
        assert self.rootPath != "", "root path is empty"
        rootname = self.rootPath + '/pts'
        
        result = []
        ext = ['csv', 'npy']

        files = self._filenames(rootname)
        for file in files:
            if file.endswith(tuple(ext)):
                result.append(file)

        return self._filenames(rootname)
    


    # 将csv转成npy文件
    # training下的三个目录中的文件名相同,分别提取出来合并成一个cvs文件
    # 将csv文件转换成与Seg相似的NPY文件
    
    # seg中npy文件格式为[x, y, z, intensity, range, category], 共有 64 * 512 = 32768个
    # 而我们的每个csv有57888个点。
    
    # 返回一个(57888*, 6)
    def cover_csv_to_nzero(self, filename, savecsv=False,
                            ptsdirname='pts', intensitydirname='intensity', categorydirname='category'):
        
        rootpath = self.rootPath
        ptsPath = self.rootPath + '/' + ptsdirname + '/' + filename
        intensityPath = self.rootPath + '/' + intensitydirname + '/' + filename
        categoryPath = self.rootPath + '/' + categorydirname + '/' + filename
        
        npypath = self.savePath + '/' + filename
        
        pts = pd.read_csv(ptsPath, header=0)
        # pts.columns = ['x', 'y', 'z', 'i', 'c']
        intensity = pd.read_csv(intensityPath)
        category = pd.read_csv(categoryPath)
        
        # print '---- pts ----'
        # print pts.ix[1]
        
        # 连接操作
        contact = pd.concat([pts, intensity, category], axis=1)
        #
        data = pd.DataFrame(contact)
        data.columns = ['x', 'y', 'z', 'i', 'c']
        data.insert(3, 'r', 0)
        
        # data.loc[:, 'r'] = 0
        # data.loc[:, ['i', 'r']] = data.loc[:, ['r', 'i']].values # 交换值
        
        # print data
        # def range(x, y, z):
        #     total = x ** 2 + y ** 2 + z ** 2
        #     # print np.sqrt(total)
        #     # np.sqrt(x ** 2 + y ** 2 + z ** 2)
        #     return cmath.sqrt(total)
        
        
        # print '----- contact -----'
        data['r'] = np.sqrt(data['x'] ** 2 + data['y'] ** 2 + data['z'] ** 2)
        
        # print data
        nzero = data.loc[(data.x != 0) & (data.y != 0) & (data.z != 0)]
        
        # print nzero
        if savecsv:
            nzero.to_csv('./contact.csv', index=False, header=False)
    
        return nzero
    
    
    def get_degree(self, x, y, z):
        
        sqrt_xy = np.sqrt(x ** 2 + y ** 2)
        # sqrt_xyz = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        
        theta = np.arctan(z / sqrt_xy) * 180 / math.pi

        # phi
        phi = np.arcsin(y / sqrt_xy) * 180 / math.pi
        
        # 调整角度
        if y > 0:
            phi = 90 + phi
        else:
            phi = phi

        # 防止越界
        if phi > 135.0:
            phi = 135.0
        elif phi < 45:
            phi = 45.0
        
        return theta, phi
        
    
    def get_point(self, theta, phi):
        
        # image x(height) * y(width) 2d
        # 向下取整
        x = int((theta - (-16)) / (32.0 / 64))
        y = int((phi - 45.0) / (90.0 / 512))
    
        # 严防越界
        x = (x > 63) and 63 or x
        y = (y > 511) and 511 or y
        
        return x, y

    def get_thetaphi(self, x, y, z):
        theta, phi = self.get_degree(x, y, z)
        
        return self.get_point(theta, phi)

    def get_point_theta(self, x, y, z):
        theta, phi = self.get_degree(x, y, z)
        
        return self.get_point(theta, phi)[0]
    
    def get_point_phi(self, x, y, z):
        theta, phi = self.get_degree(x, y, z)
        
        return self.get_point(theta, phi)[1]
    

    def isempty(self, x):
        if (x==[0, 0, 0]).all():
            return True
        else:
            return False

    
        
    # 转换成npy格式 np
    def generate_image_np(self, source, debug=False):
        
        data = source.values
        # print type(data)
        
        x = [data[i][0] for i in range(len(data[:, 0]))]
        y = [data[i][1] for i in range(len(data[:, 0]))]
        z = [data[i][2] for i in range(len(data[:, 0]))]
        intensity = [data[i][3] for i in range(len(data[:, 0]))]
        distance = [data[i][4] for i in range(len(data[:, 0]))]
        label = [data[i][5] for i in range(len(data[:, 0]))]
    
        thetaPt = [self.get_point_theta(data[i][0], data[i][1], data[i][2]) for i in range(len(data[:, 0]))] # x
        phiPt = [self.get_point_phi(data[i][0], data[i][1], data[i][2]) for i in range(len(data[:, 0]))] # y
        
        # 生成数据 phi * theta * [x, y, z, i, r, c]
        image = np.zeros((64, 512, 6), dtype=np.float16)
        
        def store_image(index):
            # print (theta[index], phi[index])
            
            image[thetaPt[index], phiPt[index], 0:3] = [x[index], y[index], z[index]]
            image[thetaPt[index], phiPt[index], 3] = intensity[index]
            image[thetaPt[index], phiPt[index], 4] = distance[index]
            image[thetaPt[index], phiPt[index], 5] = label[index]
        
        for i in range(len(x)):
            if x[i] < 0.5: continue # 前向
            if abs(y[i]) < 0.5: continue
            
            if self.isempty(image[thetaPt[i], phiPt[i], 0:3]):
                store_image(i)
            elif label[i] == image[thetaPt[i], phiPt[i], 5]:
                if distance[i] < image[thetaPt[i], phiPt[i], 4]:
                    image[thetaPt[i], phiPt[i], 4] = distance[i]
            elif image[thetaPt[i], phiPt[i], 5] == 0 and label[i] != 0:
                store_image(i)
            else:
                if distance[i] < image[thetaPt[i], phiPt[i], 4]:
                    store_image(i)
                
     
        if debug:
            
            # print theta, phi
            start = time.time()
            for i in range(len(x)):
                # print x[i], y[i], z[i], intensity[i], distance[i], label[i]
                value = x[i]
    
            print time.time() - start
    
            start = time.time()
            for i in range(len(x)):
                # print data[i]
                value = data[i]
    
            print time.time() - start
            
            print source.values
            print 'type: %s' % type(source.values)
            print np.shape(source.values)
            print np.shape(image)
        
        return image
    
    
    # 转换成需要的npy格式 for循环写法
    def generate_np_format(self, source, statistic=False):
        
        n = 0 #
        max_phi, min_phi = 0, float('inf')
        
        cCount = [0] * 8
        updateCount = [0] * 2
        

        # height x width x {x, y, z, intensity, range, label}
        npy = np.zeros((64, 512, 6), dtype=np.float16)
        
        start = time.time()
        for indexs in source.index:
            
            # 取出列表每行的值
            values = source.loc[indexs].values[:]
            x, y, z, i, r, c = values[0], values[1], values[2], values[3], values[4], values[5]
        
            if x < 0.5: continue # 前向
            if abs(y) < 0.5: continue
            
            # theta -16~16 phi 45~135
            theta, phi = self.get_degree(x, y, z)
            
            # 由x, y, z计算出的点
            ptx, pty = self.get_point(theta, phi)
            
            
            if not self.isempty(npy[ptx, pty, 0:3]): # 该点上已经有值
                
                lastpoint = npy[ptx, pty, :]
                
                if lastpoint[5] == 0: # 0表示不关心的点 category == 0
                    npy[ptx, pty, :] = [x, y, z, i, r, c]
                    updateCount[0] += 1
                    
                elif r < lastpoint[4]:
                    npy[ptx, pty, :] = [x, y, z, i, r, c]
                    updateCount[1] += 1
                    

            else:
                npy[ptx, pty, :] = [x, y, z, i, r, c]
                

            if statistic:
                if n == 0:
                    print ptx, pty
                    print 'values test: '
                    print type(values)
                    print (values)
                    print (x, y, z)
                    print theta, phi
                    n += 1
                    
                # 结果统计
                if phi > max_phi: max_phi = phi
                if phi < min_phi: min_phi = phi
                if c < 8: cCount[int(c)] += 1
        end = time.time()
        
        if statistic:
            # print 'point count is: %d' % (self._array_flag_count(xyflag, 1))
            print 'duration is %s' % (end - start)
            print 'phi max and min: %f %f' % (max_phi, min_phi)
            print 'category count statistic: %s' % cCount
            print 'update count statistic: %s' % updateCount
            
        return npy

    # 所有子文件
    def _filenames(self, filedir):
        result = []
        for root, dirs, files in os.walk(filedir):
            # print "root: {0}".format(root)
            # print "dirs: {0}".format(dirs)
            # print "files: {0}".format(files)
            result = files
        return result
    
    # 统计标记数量
    def _array_flag_count(self, array, flag):
        count = 0
        for num in array:
            if num == flag:
                count += 1
        return count
    
        
'''
if __name__ == '__main__':
    
    testpath = '../../data/training'
    
    compontent = TransformData()
    compontent.rootPath = testpath
    # print(compontent.load_file_names(testpath))

    print '正在转换......'
    result = compontent.cover_csv_to_nzero('ac3fc22d-f288-477f-a7aa-b73936b23f91_channelVELO_TOP.csv')
    print type(result)

    slow = True
    global formatdata
    
    if not slow:
        formatdata = compontent.generate_np_format(result, statistic=True)
    
    
    else:
        formatdata = compontent.generate_image_np(result, debug=True)

    print '转换后数据：'
    # 生成一个csv文件
    pdata = pd.DataFrame(np.reshape(formatdata, (-1, 6)), columns=['x', 'y', 'z', 'intensity', 'range', 'category'])
    pdata[['x', 'y', 'z', 'intensity', 'range', 'category']].astype('float64').to_csv('transnpy_quick.csv', index=None,
                                                                                      header=None)
    # 生成一个npy文件
    np.save("./data_quick.npy", formatdata)

    # 检查npy文件
    npy = np.load("./data_quick.npy")

    print '文件转换已完成！'
    print np.shape(npy)
    
    # testnp = np.zeros((2, 5, 4))
    # print testnp
    # testnp[1, 3, 3] = 9
    # print testnp
    
    
# '''