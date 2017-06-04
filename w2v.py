#encode:utf-8
import gensim
from gensim import corpora, models, similarities  
import jieba
from pprint import pprint
documents=[
'''近年来，无监督的深度神经网络在计算机视觉技术、自然语言处理和语音识别任务上都已经取得了很大的进步，而在信息检索的排序上却仍在原地踏步，没有太大的改进。其中的原因可能在于排序问题本身的复杂性，因为在没有监督信号的情况下，神经网络很难从查询内容和文档中获取信息。因此，我们在这篇文章中提出了使用“弱监督”来训练神经排序模型。也就是说，所有训练所需的标签都是机器自己获取的，不存在任何人工输入的标签。
''']

IGNORE = {'的','，','。','？','！','、','‘','’','“','”','了','\n'}

texts = [[word for word in jieba.cut(doc) if word not in IGNORE] for doc in documents]


dictionary = corpora.Dictionary(texts)
dictionary.save('test.dict') 

newString = '无监督的深度神经网络'
print(dictionary.doc2bow([word for word in jieba.cut(newString)]))



