#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:29:03 2020

@author: yojeremijenko-conel
"""

from imageai.Detection import ObjectDetection
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import timeit


os.chdir('/Users/yojeremijenko-conel/Documents/GitHub/Satelite-Research-Project/FirstTraining')
execution_path = os.getcwd()

detector = ObjectDetection()
#detector.setModelTypeAsYOLOv3()
#detector.setModelTypeAsTinyYOLOv3()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
detector.loadModel()
   
path = "Income_SDGs_MPI_Oct2019.csv"
df = pd.read_csv(path, encoding = "ISO-8859-1")
df = df[['municipality', 'commune',  'pic_1','panoramaid1','pic_2_1','panoramaid2','pic_2_2',
         'panoramaid3','pic_2_3','panoramaid4','pic_2_4','panoramaid5','pic_2_5']]

lst2 = []
problst2 = []



def find_objects(pic):
    fname = os.path.join(execution_path, 'angola_dg', pic)
    outname = os.path.join(execution_path, 'angola_dg', 'AI_' + pic)
    try:
        detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , fname),
                                             output_image_path=os.path.join(execution_path , outname))
        #print(detections)
        return detections
    except:
        return np.nan
#%%    
    
find_objects('uuid_0aac332e-4487-47d9-9b16-711f87b1d580.jpeg')



#%%
"""

sample = df.sample(400)

start = timeit.default_timer()

sample['pic_2_2_detections'] = sample.pic_2_2.apply(find_objects)

stop = timeit.default_timer()

print('Time: ', stop - start)

sample.to_csv('sample_detections.csv')
"""

#%%
"""
#start = timeit.default_timer()
#df['pic_2_2_detections'] = df.pic_2_2.apply(find_objects)
#df.to_csv('df_detections.csv')
print('pic_2_2_detections')
df['pic_1_detections'] = df.pic_1.apply(find_objects)
df.to_csv('df_detections.csv')
print('pic_1_detections')

df['pic_2_1_detections'] = df.pic_2_1.apply(find_objects)
df.to_csv('df_detections.csv')
print('pic_2_1_detections')


df['pic_2_3_detections'] = df.pic_2_3.apply(find_objects)
df.to_csv('df_detections.csv')
print('pic_2_3_detections')

df['pic_2_4_detections'] = df.pic_2_4.apply(find_objects)
df.to_csv('df_detections.csv')
print('pic_2_4_detections')

df['pic_2_5_detections'] = df.pic_2_5.apply(find_objects)
df.to_csv('df_detections.csv')
print('pic_2_5_detections')


stop = timeit.default_timer()

print('Time: ', stop - start)
"""

