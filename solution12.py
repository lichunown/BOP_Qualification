#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:14:44 2017

@author: lab309
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 14:19:36 2017

@author: lab309
"""

# CNN

import numpy as np
from readData import *

#model = gensim.models.Word2Vec.load("wiki.zh.text.model")

import keras,os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Model


RUN_TRAIN = False

qst_input = Input(shape=(200, 400, 1),dtype = 'float32',name = 'qst_input')
ans_input = Input(shape=(200, 400, 1),dtype = 'float32',name = 'ans_input')

qst_x = Flatten()(qst_input)
qst_out = Dense(500, activation='sigmoid')(qst_x)

ans_x = Flatten()(ans_input)
ans_out = Dense(500, activation='sigmoid')(ans_x)

x = keras.layers.concatenate([qst_out, ans_out])
x = Dense(1000, activation='sigmoid')(x)
x = Dense(1, activation='sigmoid')(x)

cnn_model = Model(inputs=[qst_input, ans_input], outputs=[x])
cnn_model.compile(optimizer='rmsprop', loss='binary_crossentropy',loss_weights=[0.01])

def getWeight(name = 'cnn_model.weight_50_300'):
    cnn_model.load_weights(name)
    return cnn_model



    
def testModel(model=cnn_model,data = devData,n = 1):
    x,y = next(yieldData3(devData,n))
    y_ = cnn_model.predict(x)
    return  np.concatenate((y,y_,y-y_),1),np.max(y_)


def testAllsu(data = devData,n=1):
    succeednum = 0
    testn = 0
    yielddatas = yieldData3(devData,1)
    for x,y in yielddatas:
        testn += 1
        y_ = cnn_model.predict(x)
        yti = np.where(y==1.)[0][0]
        if y_[yti,0] == np.max(y_):
            succeednum += 1
        if testn%100 ==0:print('[running] testAllsu read %d' % testn)
        if testn > n:break
    return succeednum/n
        


save_model_dir = './models'    
model_name = 'cnn_model.weight_sigmod2'

def train(per_traindata_num=50,all_iter=20,testdatas=50,loads_weight=True):
    yielddatas = yieldData3(trainData,20)
    if loads_weight:
        try:
            getWeight('cnn_model.weight_relu')
            print('[history] read model weight.')
        except Exception:
            print('[warning] Can\'t load Weight.')
    for i in range(all_iter):
        print('[running] train iter %d'%i)
        try:
            cnn_model.fit_generator(yielddatas,per_traindata_num)
        except ValueError:
            yielddatas = yieldData3(trainData,50)
            print('[Error] fuck the yield.           GG')
        cnn_model.save_weights(os.path.join(save_model_dir,'%s_%d'%(model_name,i)))    
        cnn_model.save(os.path.join(save_model_dir,'%s_%d'%(model_name,i)))
        print('[test] iter %d  ---->    succeed: %f' % (i,testAllsu(n=testdatas)))


if __name__=='__main__':
    if RUN_TRAIN:
        train()
