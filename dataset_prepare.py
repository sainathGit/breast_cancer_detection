# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:54:55 2020

@author: siddh
"""


from pathlib import Path
import numpy as np
from shutil import copyfile
import pickle as pkl
import os
from skimage.io import imread, imshow, imread_collection, concatenate_images,imsave
# for root, dirs, files in os.walk("../breast-cancer-icpr-contest"):
# #     for file in files:
        
# #         path = os.path.join(root, file)
# #         if '_m_cell' in file:     
# #             copyfile(path, str(Path('../segmented/mitotic')/file))
        
# #         elif '_nm_cell' in file:
# #             copyfile(path, str(Path('../segmented/non_mitotic')/file))
#     pass

#pickle creator
import random
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = (80, 80, 3)
basepath_1 = '../segmented/mitotic/'
basepath_2 = '../segmented/non_mitotic/'
mitotic_entries = os.listdir(basepath_1)
nmitotic_entries = os.listdir(basepath_2)

k = len(nmitotic_entries)

nm = random.sample(range(0,k),10000)

p1 = (len(mitotic_entries)*3)//4
p2 = (len(nm)*3)//4


mx_train = np.zeros((len(mitotic_entries), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
nx_train  = np.zeros((len(nm), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

for i in range(len(nm)):
    
    nx_train[i] = imread(basepath_2+nmitotic_entries[nm[i]])
    if i < len(mx_train):
        mx_train[i] = imread(basepath_1+mitotic_entries[i])
     
print('done loading')
nx_train = nx_train.astype('float32')     
mx_train = mx_train.astype('float32')        


       
pickle_out = open(str(Path("../processed_data/mitotic.pkl")),"wb")
pkl.dump((mx_train[:p1],mx_train[p1:]), pickle_out)

pickle_out = open(str(Path("../processed_data/non_mitotic.pkl")),"wb")
pkl.dump((nx_train[:p2],nx_train[p2:]), pickle_out)

pickle_out.close()
print('pickles done')

        