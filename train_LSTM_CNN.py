#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lcy
"""
#LSTM_CNN
import os
import numpy as np
from readData import yieldData,devData,trainData,testData,trueOfDataPercent

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Input,LSTM,Convolution1D,MaxPooling1D,Merge
from keras.layers import Conv2D, MaxPooling2D,Conv1D
from keras.utils.np_utils import to_categorical

os.mkdir('models/')

RUN = False

class LSTM_CNN_Model():
    def __init__(self,QA_EMBED_SIZE = 64,BATCH_SIZE = 32):
        self.QA_EMBED_SIZE = QA_EMBED_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self._model = self.createLSTMModel()
        
    def createLSTMModel(self):
        qenc = Sequential()
        qenc.add(LSTM(self.QA_EMBED_SIZE, return_sequences=True,input_shape=(200,400)))
        qenc.add(Dropout(0.3))
        qenc.add(Conv1D(self.QA_EMBED_SIZE // 2, 5, border_mode="valid"))
        qenc.add(MaxPooling1D(pool_length=2, border_mode="valid"))
        qenc.add(Dropout(0.3))
        qenc.add(Flatten())
        aenc = Sequential()
        aenc.add(LSTM(self.QA_EMBED_SIZE, return_sequences=True,input_shape=(200,400)))
        aenc.add(Dropout(0.3))
        aenc.add(Conv1D(self.QA_EMBED_SIZE // 2, 3, border_mode="valid"))
        aenc.add(MaxPooling1D(pool_length=2, border_mode="valid"))
        aenc.add(Dropout(0.3))
        aenc.add(Flatten())
        _model = Sequential()
        _model.add(Merge([qenc, aenc], mode="concat", concat_axis=-1))
        _model.add(Dense(2, activation="softmax"))
        _model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=["accuracy"])
        return _model
    
    def train(self,datas,epoch = 2,save_step=1000,savename = 'models/LSTM_CNN'): 
        tpercent = trueOfDataPercent(datas)
        tweight = 1/tpercent
        fweight = 1/(1-tpercent)
        for epoch in range(epoch):
            print('[running] train epoch %d .' % epoch)
            yielddatas = yieldData(datas,self.BATCH_SIZE,[tweight,fweight])
            tempi = 0
            while True:
                try:
                    self._model.fit_generator(yielddatas, save_step)
                    tempi += 1
                except StopIteration:
                    print('[message] %d epoch end. Train all datas %d+'%(epoch,tempi*save_step))
                    break
                except Exception:
                    print('[error]!')
                    break
            self._model.save_weights(savename+'_%d_%d'%(epoch,tempi))
            
    @property
    def model(self):
        return self._model

if __name__=='__main__' and RUN:
    lmodel = LSTM_CNN_Model()
    lmodel.train(trainData)



        
        
#==============================================================================
# def test(model=model,n=1):
#     yielddatas = yieldData3_2(trainData,1)
#     x,y = next(yielddatas)
#     z = model.predict(x)
#     print(z)
#     print(np.max(z))
# 
# def testall(weightname='models/LSTM_3_0', datas = devData,n=1000):
#     lstmmodel = LSTMModel()
#     lstmmodel.load_weights(weightname)
#     yielddatas = yieldData3_2(datas,10)
#     lstmmodel.evaluate_generator(yielddatas,n/10)
#==============================================================================
        
