#from gensim import corpora, models, similarities  
import jieba
import gensim
#from gensim.corpora import Dictionary, WikiCorpus
#from gensim.models import  Word2Vec

"""

model = gensim.models.Word2Vec.load("wiki.zh.text.model")
#from readData import trainData,devData

documents = trainData[0][1]

IGNORE = {'的','，','。','？','！','、','‘','’','“','”','了','\n'}

texts = [[word for word in jieba.cut(doc) if word not in IGNORE] for doc in documents]

dictionary = corpora.Dictionary(texts)
dictionary.save('test.dict') 


newString = trainData[0][0]


new_v = dictionary.doc2bow([word for word in jieba.cut(newString)])
print(new_v)


corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize('tempcorpus.mm', corpus)
corpus = corpora.MmCorpus('tempcorpus.mm')
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

def sentenceCut(sen):
    return [i for i in jieba.cut(sen)]"""
"""  
data = devData[0]

question = data[0]
answers = data[1]
output = data[2]

questionlist = [item for item in jieba.cut(question)]
answerslists = []
for sen in answers:
    answerslists.append([item for item in jieba.cut(sen)])"""
    
#http://www.cnblogs.com/iloveai/p/gensim_tutorial.html     
             
from gensim import corpora
import codecs
from gensim import models
from gensim import similarities

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
    return lsi_model[s]
