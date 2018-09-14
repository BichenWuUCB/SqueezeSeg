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
    elif category == 7:
        return 255, 128, 0
    else :
        return 0,0,0
    
    

zipfilename='dataset.zip'
count=0
azip = zipfile.ZipFile(zipfilename)  # ['bb/', 'bb/aa.txt']
for file in azip.namelist():
    if(os.path.splitext(file)[1] =='.csv'):
        if('category' in file):
            ptsfilename=file.replace('category','pts')
            pts=azip.read(ptsfilename).decode('utf-8').replace('\n',',').replace(',',' ')
            intensityfilename=file.replace('category','intensity')
            intens=np.fromstring(azip.read(intensityfilename).decode('utf-8').replace('\n',','),dtype=float,sep=',')            
            lablefilename=file
            lable=np.fromstring(azip.read(lablefilename).decode('utf-8').replace('\n',','),dtype=float,sep=',')
            data=np.fromstring(pts, dtype=float, sep=' ')
            k=int(len(data)/3)
            data=data.reshape(k,3)
            color=[get_color(lable[i]) for i in range(len(data[:,0]))]
            
            txtfile=np.hstack((data,color))
            if(os.path.exists(os.path.dirname(ptsfilename))==False):
                os.makedirs(os.path.dirname(ptsfilename))
            np.savetxt(ptsfilename.replace('.csv','.txt'), txtfile, delimiter=' ')
            count=count+1
            print(count)      
           
            
         
print ('OK!')