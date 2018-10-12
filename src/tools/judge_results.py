#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:18-9-26 下午3:03
# software:PyCharm

# alibaba

import csv
import pandas as pd

import zipfile

def judge(standardResults, userCommitFile):
    
    result = [0.0] * 9
    
    num_pred = [0] * 8
    result_pred = [0.0] * 8
    
    pred_files = zipfile.ZipFile(userCommitFile)
    for filename in pred_files.filename:
        
        eval_filename = filename[0:-4]
        
        gt = get_results(standardResults + filename)
        pred = get_results(userCommitFile + filename)
        
        if gt.shape != pred.shape:
            result[0] = 2.0
            break
        
        union = [0] * 8
        intersection = [0] * 8
        
        for i in range(8):
            continue
    


def get_results(filepath, debug=False):
    
    ext = ['csv']
    assert filepath.endswith(tuple(ext)), "file format error"
    
    label_info = [0]
    
    if debug:
        with csv.reader(filepath, 'r') as csv_file:
            for data in csv_file:
                print data
    else:
        df = pd.read_csv(filepath, header=None)
        data = df.values()
        
    return data
    
   
    
    
    
    

if __name__ == "__main__":
    print "generate test!"
    
    
    