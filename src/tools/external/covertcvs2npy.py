#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 10:24:55 2018

@author: mengwl
"""

import zipfile
#import csv
import os
#import os.path 
import numpy as np
#import string

zipfilename='train_fix.zip'
azip = zipfile.ZipFile(zipfilename)  # ['bb/', 'bb/aa.txt']
for file in azip.namelist():
    if(os.path.splitext(file)[1] =='.csv'):
        if('pts' in file):
            pts=azip.read(file).decode('utf-8').replace('\n',',').replace(',',' ')
            intensityfilename=file.replace('pts','intensity')
            intens=np.fromstring(azip.read(intensityfilename).decode('utf-8').replace('\n',','),dtype=float,sep=',')            
            lablefilename=file.replace('pts','category')
            lable=np.fromstring(azip.read(lablefilename).decode('utf-8').replace('\n',','),dtype=float,sep=',')
            data=np.fromstring(pts, dtype=float, sep=' ')
            k=int(len(data)/3)
            data=data.reshape(k,3)
            dis=np.linalg.norm(data,axis=1)
            data=np.c_[data,intens,dis,lable]
            len(data)
            savefilename=file.replace('pts','npy')
            savefilename=savefilename.replace('csv','npy')
            if(os.path.exists(os.path.dirname(savefilename))==False):
                os.makedirs(os.path.dirname(savefilename))
            np.save(savefilename,data)
            
         
print ('OK!')