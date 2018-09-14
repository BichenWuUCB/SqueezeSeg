import numpy as np
import os

print "start:"
print os.getcwd()
# print os.path.abspath('print.py')

out = np.load("./data/samples_out/pred_2011_09_26_0001_0000000010.npy")
print out
out = np.load("./data/samples_out/pred_2011_09_26_0001_0000000050.npy")
print out

# /Users/dizi/Git/casia/SqueezeSeg/src/print.py