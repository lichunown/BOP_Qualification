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

#from readData import *
import numpy as np
import gensim
from readData import *

#model = gensim.models.Word2Vec.load("wiki.zh.text.model")

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Model




qst_input = Input(shape=(200, 400, 1),dtype = 'float32',name = 'qst_input')
ans_input = Input(shape=(200, 400, 1),dtype = 'float32',name = 'ans_input')

qst_x = Conv2D(32, (3, 3), activation='relu')(qst_input)
qst_x = MaxPooling2D(pool_size=(2, 2))(qst_x)
qst_x = Dropout(0.05)(qst_x)
qst_x = Flatten()(qst_x)
qst_out = Dense(2000, activation='relu')(qst_x)

ans_x = Conv2D(32, (3, 3), activation='relu')(qst_input)
ans_x = MaxPooling2D(pool_size=(2, 2))(ans_x)
ans_x = Dropout(0.05)(ans_x)
ans_x = Flatten()(ans_x)
ans_out = Dense(2000, activation='relu')(ans_x)

x = keras.layers.concatenate([qst_out, ans_out])
x = Dense(4000, activation='sigmoid')(x)
x = Dense(1, activation='sigmoid')(x)

cnn_model = Model(inputs=[qst_input, ans_input], outputs=[x])
cnn_model.compile(optimizer='rmsprop', loss='binary_crossentropy',loss_weights=[0.01])



yielddatas = yieldData(10)
cnn_model.fit_generator(yielddatas,1000)
cnn_model.save('cnn.model')
cnn_model.save_weights('cnn_model.weight')
#predict







