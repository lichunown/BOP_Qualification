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



"""
x_train = np.random.random((data_nums, 100, 400, 1))
y_train = keras.utils.to_categorical(np.random.randint(1, size=(data_nums, 1)), num_classes=2)


x_test = np.random.random((20, 100, 400, 1))
y_test = keras.utils.to_categorical(np.random.randint(1, size=(20, 1)), num_classes=2)


qst_train,ans_train,tof_train = datas2vec2(trainData)


cnn_model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.

cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Flatten())
cnn_model.add(Dense(256, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
cnn_model.compile(loss='categorical_crossentropy', optimizer=sgd)

cnn_model.fit(x_train, y_train, batch_size=32, epochs=1)
score = cnn_model.evaluate(x_test, y_test, batch_size=32)"""
##############################

qst_input = Input(shape=(200, 400, 1),dtype = 'float32',name = 'qst_input')
ans_input = Input(shape=(200, 400, 1),dtype = 'float32',name = 'ans_input')

qst_x = Conv2D(32, (3, 3), activation='relu')(qst_input)
qst_x = Conv2D(32, (3, 3), activation='relu')(qst_x)
qst_x = MaxPooling2D(pool_size=(2, 2))(qst_x)
qst_x = Dropout(0.25)(qst_x)
qst_x = Flatten()(qst_x)
qst_out = Dense(1024, activation='relu')(qst_x)

ans_x = Conv2D(32, (3, 3), activation='relu')(qst_input)
ans_x = Conv2D(32, (3, 3), activation='relu')(ans_x)
ans_x = MaxPooling2D(pool_size=(2, 2))(ans_x)
ans_x = Dropout(0.25)(ans_x)
ans_x = Flatten()(ans_x)
ans_out = Dense(1024, activation='relu')(ans_x)

x = keras.layers.concatenate([qst_out, ans_out])
x = Dense(4000, activation='sigmoid')(x)
x = Dense(1, activation='sigmoid')(x)

cnn_model = Model(inputs=[qst_input, ans_input], outputs=[x])
cnn_model.compile(optimizer='rmsprop', loss='binary_crossentropy',loss_weights=[0.01])

checkpointer = keras.callbacks.ModelCheckpoint(filepath='cnn_weight.hdf5',verbose=1,save_best_only=True)

#qst_input_data = np.random.random((data_nums, 100, 400, 1))
#ans_input_data = np.random.random((data_nums, 100, 400, 1))
#output_data = keras.utils.to_categorical(np.random.randint(1, size=(data_nums, 1)), num_classes=1)

#cnn_model.fit([qst_input_data, ans_input_data], [output_data],epochs=1, batch_size=32)

batch = 100
for i in range(50):
    print("train data:  %d ~ %d "% (batch*i,batch*i+batch))
    datas = getRandomData(trainData,batch)
    qst_input_data,ans_input_data,output_data = datas2vec3(datas,200)
    del datas
    cnn_model.fit([qst_input_data, ans_input_data], [output_data],epochs=1, batch_size=32,
                  validation_split=0.05,callbacks=[checkpointer])
    print('save--%d'%(i+1))
    


datas = getRandomData(devData)
x_test,x_test2,y_test = datas2vec2(datas)

score = cnn_model.evaluate([x_test,x_test2], [y_test], batch_size=32)




