#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:25:17 2020

@author: yojeremijenko-conel
"""


from imageai.Detection import ObjectDetection
import os
import tensorflow as tf
import pandas as pd
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
fname = 'sidney2.jpg'
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , fname),
                                             output_image_path=os.path.join(execution_path , 'sidney2_ai.jpg'))

for eachObject in detections:
   print(eachObject["name"] , " : " , eachObject["percentage_probability"])
    
