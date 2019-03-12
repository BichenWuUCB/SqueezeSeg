#!/usr/bin/env python

import os, zipfile

def zip_answers(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join('./', filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)
            
            zipf.write(pathfile, arcname)
    zipf.close()


if __name__ == '__main__':
    source_dir = '/home/mengweiliang/lzh/SqueezeSeg/scripts/log/answers/'
    output_filename = 'answers.zip'
    zip_answers(source_dir, output_filename)

    
