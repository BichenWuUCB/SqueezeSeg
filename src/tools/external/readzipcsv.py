#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 14:40:16 2018

@author: mengwl
"""

import zipfile
#import csv
import os.path 
import numpy as np
#import string

azip = zipfile.ZipFile('dataset.zip')  # ['bb/', 'bb/aa.txt']
for file in azip.namelist():
    if(os.path.splitext(file)[1] =='.csv'):
        if('pts' in file):
            pts=azip.read(file).decode('utf-8').replace('\n',',').replace(',',' ')
            intensityfilename=file.replace('pts','intensity')
            intens=np.fromstring(read(intensityfilename).decode('utf-8').replace('\n',','),dtype=float,sep=' ')            
            lablefilename=file.replace('pts','category')
            lable=np.fromstring(azip.read(lablefilename).decode('utf-8').replace('\n',','),dtype=float,sep=' ')
            data=np.fromstring(pts, dtype=float, sep=' ')
            k=int(len(data)/3)
            data=data.reshape(k,3)
           

#