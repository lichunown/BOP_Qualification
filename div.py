#decode:utf-8
from readData import trainData,devData
import jieba
import jieba.posseg as pseg
import numpy as np
import codecs


fname = 'wiki.zh.text.jian'
writename = 'wiki.zh.text.jian.div'


f = codecs.open(fname,'r')
output = codecs.open(writename,'w')
i=0
for line in f:
    linelist = [item for item in jieba.cut(line) if item!=' ']
    output.write(' '.join(linelist))
    output.write('\n')
    i += 1
    if i%100==0:
        print('write line %d' % i)

f.close()
output.close()



