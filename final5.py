# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 23:41:58 2017

@author: lichunyang
"""

import codecs
from solution30 import *

lstmmodel = LSTMModel()
lstmmodel.load_weights('models/LSTM_3_6')


def readFinalDataFromTxt(filename,decode='utf8'):
    f = codecs.open(filename,'r',decode)
    data = []
    for line in f:
        data.append([string.replace('\r\n','') for string in line.split('\t')])
    return data
        
finalData=readFinalDataFromTxt(os.path.join(dataDir,'BoP2017-DBQA.test.txt'))

writefile = open('final/finaldata6.txt','w')
i = 0
for data in finalData:
    x = [sts2vec3(data[0],200).reshape(1,200,400),sts2vec3(data[1],200).reshape(1,200,400)]
    i+=1
    r = lstmmodel.predict(x)
    writefile.write(str(r[0][0]))
    writefile.write('\r\n')
    if i%100==0:
        print('write %d'%i)
writefile.close()

    
