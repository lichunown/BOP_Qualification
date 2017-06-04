#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 14:19:36 2017

@author: lab309
"""
from gensim import corpora
import codecs
from gensim import models
from gensim import similarities
import numpy as np

def sts2list(sts):
    return [item for item in jieba.cut(sts)]

#model = gensim.models.Word2Vec.load("wiki.zh.text.model")

def run(data):
    data = data
    question = data[0]
    answers = data[1]
    trueresult = data[2]
    
    question = sts2list(question)
    answers = [sts2list(item) for item in answers]
    
    questionVec = np.zeros(400,'float32')
    for i in question:
        try:
            questionVec += model[i]
        except KeyError:
            pass
            #print('word "%s" not in dictionary'%i)
    
    answerslen = len(answers)
    ansVecs = []
    for i in range(answerslen):
        ansVec = np.zeros([400],'float32')
        for word in answers[i]:
            try:
                ansVec += model[word]
            except KeyError:
                pass
                #print('word "%s" not in dictionary'%word)
        ansVecs.append(ansVec)
        
    def cos(vec1,vec2):
        return np.sum(vec1*vec2)/((np.sqrt(np.sum(vec1**2)))*(np.sqrt(np.sum(vec2**2))))
    
    result = np.zeros(len(ansVecs),'float32')
    for i in range(len(ansVecs)):
        result[i] = cos(questionVec,ansVecs[i])
    resultmax = np.argmax(result)
    trueresutmax = np.argmax(trueresult)
    if resultmax==trueresutmax:
        tof = True
    else:
        tof=False
    return result,tof,resultmax,trueresutmax


nums = len(trainData)
succeed = 0
i = 0
for data in devData:
    i+=1
    result,tof,resultmax,trueresutmax = run(data)
    if tof:
        succeed+=1
    if i%200==0:
        print("run %d"%i)
print(succeed/nums)

