#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 18:10:58 2017

@author: lab309
"""

#LSTM


import numpy as np
from readData import *

import keras,os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Input,LSTM,Convolution1D,MaxPooling1D,Merge
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Model
from keras.utils.np_utils import to_categorical

def LSTMModel():
    QA_EMBED_SIZE = 64
    BATCH_SIZE = 32
    NBR_EPOCHS = 20
    qenc = Sequential()
    qenc.add(LSTM(QA_EMBED_SIZE, return_sequences=True,input_shape=(200,400)))
    qenc.add(Dropout(0.3))
    qenc.add(Convolution1D(QA_EMBED_SIZE // 2, 5, border_mode="valid"))
    qenc.add(MaxPooling1D(pool_length=2, border_mode="valid"))
    qenc.add(Dropout(0.3))
    qenc.add(Flatten())

    aenc = Sequential()
    aenc.add(LSTM(QA_EMBED_SIZE, return_sequences=True,input_shape=(200,400)))
    aenc.add(Dropout(0.3))
    aenc.add(Convolution1D(QA_EMBED_SIZE // 2, 3, border_mode="valid"))
    aenc.add(MaxPooling1D(pool_length=2, border_mode="valid"))
    aenc.add(Dropout(0.3))
    aenc.add(Flatten())
    
    model = Sequential()
    model.add(Merge([qenc, aenc], mode="concat", concat_axis=-1))
    model.add(Dense(2, activation="softmax"))
    
    model.compile(optimizer="adam", loss='categorical_crossentropy',#loss="categorical_crossentropy",sparse_categorical_crossentropy
                  metrics=["accuracy"])
    return model


#yielddatas = yieldData3_2(trainData,10)
#x,y = next(yielddatas)
#model.fit(x, y, batch_size=BATCH_SIZE,
#          nb_epoch=NBR_EPOCHS, validation_split=0.1)


def train(model = model,temp_iter=20,savename='models/LSTM_4'):
    for i in range(temp_iter):
        print('train iter %d :'%i)
        yielddatas = yieldData3_2(trainData,10)
        try:
            model.fit_generator(yielddatas, 500)
        except Exception:
            print('[error] : I don\'t know why.')
        model.save_weights(savename+'_%d'%i)
        model.save(savename+'_%d'%i)
    return model
        
        
def test(model=model,n=1):
    yielddatas = yieldData3_2(trainData,1)
    x,y = next(yielddatas)
    z = model.predict(x)
    print(z)
    print(np.max(z))

def testall(weightname='models/LSTM_3_0', datas = devData,n=1000):
    lstmmodel = LSTMModel()
    lstmmodel.load_weights(weightname)
    yielddatas = yieldData3_2(datas,10)
    lstmmodel.evaluate_generator(yielddatas,n/10)
        
    

RUN = False
if RUN:
    model = LSTMModel()
    model = train(model)