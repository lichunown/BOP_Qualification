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
#from readData import *

#model = gensim.models.Word2Vec.load("wiki.zh.text.model")

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Model
import tensorflow as tf
from keras.engine.topology import Layer
from keras import backend as K  
'''

'''


qst_input = Input(shape=(200, 400, 1),dtype = 'float32',name = 'qst_input')
tans_input = Input(shape=(200, 400, 1),dtype = 'float32',name = 'tans_input')
fans_input = Input(shape=(200, 400, 1),dtype = 'float32',name = 'fans_input')


qst_x = Conv2D(64, (5, 5), activation='sigmoid')(qst_input)
qst_x = MaxPooling2D(pool_size=(2, 2))(qst_x)
qst_x = Dropout(0.05)(qst_x)
qst_x = Flatten()(qst_x)
qst_out = Dense(1000, activation='sigmoid')(qst_x)

tans_x = Conv2D(64, (5, 5), activation='sigmoid')(tans_input)
tans_x = MaxPooling2D(pool_size=(2, 2))(tans_x)
tans_x = Dropout(0.05)(tans_x)
tans_x = Flatten()(tans_x)
tans_out = Dense(1000, activation='sigmoid')(tans_x)

fans_x = Conv2D(64, (5, 5), activation='sigmoid')(fans_input)
fans_x = MaxPooling2D(pool_size=(2, 2))(fans_x)
fans_x = Dropout(0.05)(fans_x)
fans_x = Flatten()(fans_x)
fans_out = Dense(1000, activation='sigmoid')(fans_x)


def mloss(y_true, y_pred):
    pass



cnn_model = Model(inputs=[qst_input, tans_input, fans_input], outputs=[result_out])

cnn_model.compile(optimizer='rmsprop', loss='binary_crossentropy',loss_weights=[0.003])

"""

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

"""





