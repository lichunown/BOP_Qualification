#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:10:59 2017

@author: lab309
"""

from gensim import corpora
import codecs
from gensim import models
from gensim import similarities
import numpy as np
import jieba

def sts2list(sts):
    return [item for item in jieba.cut(sts)]
def run(data):
    data = data
    question = data[0]
    answers = data[1]
    result = data[2]
    anslist = []
    for line in answers:
        anslist.append(sts2list(line))
    dictionary = corpora.Dictionary(anslist)
    corpus = [dictionary.doc2bow(text) for text in anslist]
    lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    documents = lsi_model[corpus]
    
    quelist = [sts2list(question)]
    query = [dictionary.doc2bow(text) for text in quelist]
    query_vec = lsi_model[query]
    
    index = similarities.MatrixSimilarity(documents)
    
    runresult = np.argmax(index[query_vec])
    expectresult = np.where(result==1)
    if runresult==expectresult:
        succeed=True
    else:
        succeed=False
    return index[query_vec].T,succeed

"""   

fname = 'wiki.zh.text.jian.div'
f = codecs.open(fname,'r',encoding='utf8')
texts = []
i=0
for line in f:
    i += 1
    texts.append(line.split())
    if i%500==0:
        print('read lines %d'%i)

print('Read END')
print('-----------------------------')

dictionary = corpora.Dictionary(texts)
print('dictionary OK')
print('-----------------------------')
corpus = [dictionary.doc2bow(text) for text in texts]
print('corpus OK')
print('-----------------------------')
corpora.MmCorpus.serialize('wikicorpus.mm', corpus)
print('write corpus "wikicorpus.mm"   OK')
print('-----------------------------')
lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
print('create lsi_model    OK')
print('-----------------------------')
documents = lsi_model[corpus]
#query_vec = lsi_model[query]
index = similarities.MatrixSimilarity(documents)
print('similarities.MatrixSimilarity    OK')
print('-----------------------------')

index.save('/tmp/wikidocuments.index')
print('savs wikidocuments.index    OK')
print('-----------------------------')
#index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')

def sts2list(sts):
    return [item for item in jieba.cut(sts)]
def query_vec(string):
    stringlist = sts2list(string)
    s = [dictionary.doc2bow(text) for text in stringlist]
    return lsi_model[s]"""