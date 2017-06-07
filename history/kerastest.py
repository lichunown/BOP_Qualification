#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:12:26 2017

@author: lab309
"""
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Model
import tensorflow as tf
# Generate dummy data
x1_train = np.random.random((100, 30, 30, 1))
x2_train = np.random.random((100, 30, 30, 1))
y_train = np.ones([100,1])#np.random.randint(0,2,[100,1])

input1 = Input(shape=(30, 30, 1),dtype = 'float32',name = 'input1')
input2 = Input(shape=(30, 30, 1),dtype = 'float32',name = 'input2')

qst_x = Conv2D(32, (3, 3), activation='relu')(input1)
qst_x = Conv2D(32, (3, 3), activation='relu')(qst_x)
qst_x = MaxPooling2D(pool_size=(2, 2))(qst_x)
qst_x = Dropout(0.25)(qst_x)
qst_x = Flatten()(qst_x)
qst_out = Dense(100, activation='relu')(qst_x)

ans_x = Conv2D(32, (3, 3), activation='relu')(input2)
ans_x = Conv2D(32, (3, 3), activation='relu')(ans_x)
ans_x = MaxPooling2D(pool_size=(2, 2))(ans_x)
ans_x = Dropout(0.25)(ans_x)
ans_x = Flatten()(ans_x)
ans_out = Dense(100, activation='relu')(ans_x)

x = keras.layers.concatenate([qst_out, ans_out])
x = Dense(100, activation='sigmoid')(x)
x = Dense(1, activation='sigmoid')(x)

cnn_model = Model(inputs=[input1, input2], outputs=[x])

cnn_model.compile(optimizer='rmsprop', loss='binary_crossentropy',loss_weights=[0.01])

def yieldData(nums=1):
    while True:
        yield ([np.random.random((nums, 30, 30, 1)),np.random.random((nums, 30, 30, 1))],np.random.randint(0,2,[nums,1]))
        

cnn_model.fit([x1_train,x2_train],[y_train],epochs=1, batch_size=32)