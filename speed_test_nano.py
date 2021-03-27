# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 03:57:16 2021

@author: Zephyr
"""

import numpy as np
import tensorflow as tf
from time import time

model_conv = tf.keras.models.load_model('./conv/saved_model/')
model_hwt = tf.keras.models.load_model('./wht/saved_model/')
model_fwht_tensordot = tf.keras.models.load_model('./fwht_tensordot/saved_model/')

lite_model_conv = tf.lite.Interpreter(model_path="./conv/lite_model.tflite")
lite_model_hwt = tf.lite.Interpreter(model_path="./wht/lite_model.tflite")
lite_model_fhwt_tensordot = tf.lite.Interpreter(model_path="./fwht_tensordot/lite_model.tflite")

input_shape = model_conv.input.shape.as_list()
input_size = input_shape[-2]
num_features = input_shape[-1]
num = 10

while True:
    x = np.random.rand(num, input_size, input_size, num_features).astype(np.float32)
    yy = np.zeros((num, input_size, input_size, num_features)).astype(np.float32)
        
    start = time()
    for i in range(num):
        lite_model_conv.allocate_tensors()
        lite_model_conv.set_tensor(lite_model_conv.get_input_details()[0]['index'], x[i:i+1, :, :])
        lite_model_conv.invoke()
        yy[i:i+1, :, :] = lite_model_conv.get_tensor(lite_model_conv.get_output_details()[0]['index'])
    end = time()
    time2 = end-start
    
    print('1x1 Conv')
    print('TFLite took', time2, 'S.')
    
    

    start = time()
    for i in range(num):
        lite_model_hwt.allocate_tensors()
        lite_model_hwt.set_tensor(lite_model_hwt.get_input_details()[0]['index'], x[i:i+1, :, :])
        lite_model_hwt.invoke()
        yy[i:i+1, :, :] = lite_model_hwt.get_tensor(lite_model_hwt.get_output_details()[0]['index'])
    end = time()
    time2 = end-start
    
    print('wht')
    print('TFLite took', time2, 'S.')

    
    start = time()
    for i in range(num):
        lite_model_fhwt_tensordot.allocate_tensors()
        lite_model_fhwt_tensordot.set_tensor(lite_model_fhwt_tensordot.get_input_details()[0]['index'], x[i:i+1, :, :])
        lite_model_fhwt_tensordot.invoke()
        yy[i:i+1, :, :] = lite_model_fhwt_tensordot.get_tensor(lite_model_fhwt_tensordot.get_output_details()[0]['index'])
    end = time()
    time2 = end-start
    
    print('fwht_tensordot')
    print('TFLite took', time2, 'S.')
