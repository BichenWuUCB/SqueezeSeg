#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:25:55 2018

@author: mengwl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 10:24:55 2018

@author: mengwl
"""

import zipfile
#import csv
import math
import os
#import os.path 
import numpy as np
#import string
from scipy.misc import imsave
import cv2


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
def get_sign(x):
    if x<0:
        return 2
    else:
        return 1
def isempty(x):
    if((x==[0,0,0]).all()):
        return True
    else:
        return False
    
def generateimage(x,y,intens,dis,category) :
    image = np.zeros((32, 2250,6), dtype=np.float)#水平方向360度，0。16度一格，垂直方向32度，1度一格
    count0=0
    count1=0
    count2=0
    count3=0
    count4=0
    
    for i in range(len(x)):
        if(isempty(image[y[i],x[i],0:3])):
          image[y[i],x[i],0:3]=get_color(category[i])
          image[y[i],x[i],3]=intens[i]
          image[y[i],x[i],4]=dis[i]
          image[y[i],x[i],5]=category[i]
          count2=count2+1
        elif(category[i]==image[y[i],x[i],5]):
            count0=count0+1
            if(dis[i]<image[y[i],x[i],4]):
                image[y[i],x[i],4]=dis[i]    
        elif(category[i]==0.0 and image[y[i],x[i],5]!=0.0):
            count3=count3+1
        elif(image[y[i],x[i],5]==0.0 and category[i]!=0.0):
            image[y[i],x[i],0:3]=get_color(category[i])
            image[y[i],x[i],3]=intens[i]
            image[y[i],x[i],4]=dis[i]
            image[y[i],x[i],5]=category[i]
            count4=count4+1
            
        else:
            if(dis[i]<image[y[i],x[i],4]):
                image[y[i],x[i],0:3]=get_color(category[i])
                image[y[i],x[i],3]=intens[i] 
                image[y[i],x[i],4]=dis[i] 
                image[y[i],x[i],5]=category[i]
            print(category[i],image[y[i],x[i],5])
            count1=count1+1
    return count0,count1,count2,count3,count4,image         
              
      
    
    

SAVED_FOLDER = '../data'
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
            #color=[get_color(lable[i]) for i in range(len(data[:,0]))]
            
            #txtfile=np.hstack((data,color))
            #np.savetxt(file.replace('.csv','.txt'), txtfile, delimiter=',')
            
            
            #print(np.max(data,axis=0))
            #以下计算点的纵坐标
            l1= [np.sqrt(data[i,0]*data[i,0]+data[i,1]*data[i,1]) for i in range(len(data[:,0]))]
            theta=[math.atan2(data[i,2],l1[i])/math.pi * 180 for i in range(len(data[:,0]))]
            mintheta=np.min(theta)
            steptheta=1
            #n
            theta=[math.floor((theta[i]-mintheta)/steptheta) for i in range(len(theta))]                      
            #目前根据theta可以确定纵坐标。   
            
            #以下开始计算横坐标
            stepdelta=0.16
            delta=[math.floor(math.acos(data[i,1]/l1[i])/math.pi * 180*(get_sign(data[i,1]))/stepdelta) for i in range(len(data[:,0]))]
            #到此，每个点的横坐标算完。现在要寻找是否有重复的值。
            
#            matrix=[delta,theta]
            #test code
#            mylist=[(delta[i],theta[i])for i in range(len(theta))]
#            newone=list(set(mylist))
#            mylist2=[(delta[i],theta[i],lable[i])for i in range(len(theta))]
#            newtwo=list(set(mylist2))
            
            
           
            
            dis=np.linalg.norm(data,axis=1)
            
            count0,count1,count2,count3,count4,image=generateimage(delta,theta,intens*255,dis,lable)
            cv2.imwrite(intensityfilename.replace('.csv','.png'), image[:,:,3])
            imsave(lablefilename.replace('.csv','.png'), image[:,:,0:3])
            
#            data=np.c_[data,intens,dis,lable]
#            len(data)
#            savefilename=file.replace('pts','npy')
#            savefilename=savefilename.replace('csv','npy')
#            if(os.path.exists(os.path.dirname(savefilename))==False):
#                os.makedirs(os.path.dirname(savefilename))
#            np.save(savefilename,data)
            
         
print ('OK!')