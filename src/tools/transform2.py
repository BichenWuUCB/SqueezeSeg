#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:18-10-11 下午5:11
# software:PyCharm


# transform training data
def transform_training_npy(rootpath="", debug=False):
    print rootpath
    
    trantool = ct.InputData()
    trantool.rootPath = rootpath
    trantool.savePath = "npy360"
    
    ptsfiles = trantool.load_file_names()
    
    filesname_savepath = "../../scripts/log/filenames.txt"
    with open(filesname_savepath, 'w') as f:
        for i in range(0, len(ptsfiles)):
            context = ptsfiles[i] + '\n'
            f.write(context)
        f.close()
    
    idx = 0
    NUM_CLASS = 8
    for file in ptsfiles:
        # print '正在转换 file ：%s  ......' % file
        idx += 1
        if file[-4:] == '.csv':
            prename = 'channelVELO_TOP_0000_%05d' % idx
            npyname = (prename + '.npy')
            
            data = trantool.cover_csv_to_nzero(file, savecsv=False)
            formatdata = trantool.generate_image_np360(data.values)
            
            # npy store
            if True:
                npypath = os.path.join(trantool.savePath, npyname)
                if os.path.exists(npypath):
                    continue
                np.save(npypath, formatdata)
            
            # csv store
            if False:
                csvpath = os.path.join("./", "image_csv360", file)
                if not os.path.exists(csvpath):
                    print("生成文件检查：")
                    print(csvpath)
                    pddata = pd.DataFrame(np.reshape(formatdata, (32768, 6)), \
                                          columns=['x', 'y', 'z', 'intensity', 'range', 'category'])
                    savedata = pddata[['x', 'y', 'z', 'intensity', 'range', 'category']].astype('float32')
                    savedata.to_csv(csvpath)
            
            if debug:
                if np.shape(formatdata) == (64, 512, 6):
                    print '%s 已生成' % npypath

if __name__ == '__main__':

    path = "/home/mengweiliang/disk15/df314/training"