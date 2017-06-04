#encoding:utf8
import os,codecs
import numpy as np
import jieba
import random


READMODEL = False

try:
    model
except: 
    READMODEL = True

dataDir = './BoP2017_DBAQ_dev_train_data'
trainFilename = 'BoP2017-DBQA.train.txt.new'
devFilename = 'BoP2017-DBQA.dev.txt.new'
decode = 'utf8'


IGNOREWARNING = True

def getLabel(data):
    size = len(data)
    result = np.zeros([size,1])
    i=0
    for line in data:
        result[i,0] = int(line[0].strip('\ufeff'))
        i += 1
    return result

"""
    [question,[answers],[results]]

"""
def readDataFromTxt(filename,decode='utf8'):
    f = codecs.open(filename,'r',decode)
    line = f.readline()
    returnresult = []
    data = line.split('\t')
    question = data[1]
    answers = [data[2]]
    results = [data[0]]
    while line:
        data = line.split('\t')
        line = f.readline()
        if data[1] == question:
            answers.append(data[2][0:-2])
            results.append(int(data[0].strip('\ufeff')))
        else:
            returnresult.append([question,answers,results])
            question = data[1]
            answers = [data[2][0:-2]]
            results = [int(data[0].strip('\ufeff'))]     
    return returnresult


trainData = readDataFromTxt(os.path.join(dataDir,trainFilename))
devData = readDataFromTxt(os.path.join(dataDir,devFilename))

def sts2list(sts):
    IGNORE = {'(',')','.','\,','/','\\','?','<','>','\'','[',']','"','-',
              '_','%','#','@','!','$','^','&','*','+','=',
              '。','，','！','《','》','“','”','‘','’','：','；','【',
              '】','、','——','…',''
              }
    return [item for item in jieba.cut(sts) if item not in IGNORE]



if READMODEL:
    import gensim
    model = gensim.models.Word2Vec.load("wiki.zh.text.model")
    
def sts2vec(sts):#生成1行400列的矩阵，是将所以词向量相加
    sts = sts2list(sts)
    vec = np.zeros(400,'float32')
    for word in sts:
        try:
            vec += model[word]
        except KeyError:
            pass
    return vec

def sts2vec2(sts,max_sequence_len=400):#return m*max_sequence_len 矩阵，词不在字典则添加0向量
    sts = sts2list(sts)
    vec = np.zeros([max_sequence_len,400],'float32')    
    i = 0
    for word in sts:
        try:
            vec[i,:] = model[word]
        except KeyError:
            vec[i,:] = np.zeros(400,'float32')
        i += 1
        if i >= max_sequence_len:
            if not IGNOREWARNING:
                print('sequence too long:%d'%len(sts))
            break
            #raise KeyError('sequence too long:%d'%len(sts))
    return vec

def sts2vec3(sts,max_sequence_len=400):# #return m*max_sequence_len 矩阵，词不在字典则忽略
    sts = sts2list(sts)
    vec = np.zeros([max_sequence_len,400],'float32')    
    i = 0
    for word in sts:
        try:
            vec[i,:] = model[word]
            i += 1
        except KeyError:
            pass      
        if i >= max_sequence_len:
            if not IGNOREWARNING:
                print('sequence too long:%d'%len(sts))
            break
            #raise KeyError('sequence too long:%d'%len(sts))
    return vec

def datas2vec2(datas,max_sequence_len=400):
    length = 0
    for data in datas:
        length += len(data[1])
    qstresult = np.zeros([length,max_sequence_len,400,1],'float32')
    ansresult = np.zeros([length,max_sequence_len,400,1],'float32')
    tofresult = np.zeros([length,1],'float32')
    i = 0
    for data in datas:
        j = 0
        for ans in data[1]:
            qstresult[i,:,:,0] = sts2vec2(data[0],max_sequence_len)
            ansresult[i,:,:,0] = sts2vec2(ans,max_sequence_len)
            if isinstance(data[2][j],int):
                tofresult[i,0] = float(data[2][j])
            else:
                tofresult[i,0] = float(int(data[2][j].strip('\ufeff')))
            j += 1
            i += 1
    return (qstresult,ansresult,tofresult)


def datas2vec3(datas,max_sequence_len=200):
    length = 0
    for data in datas:
        length += 2*len(data[1])-2
    qstresult = np.zeros([length,max_sequence_len,400,1],'float32')
    ansresult = np.zeros([length,max_sequence_len,400,1],'float32')
    tofresult = np.zeros([length,1],'float32')
    i = 0
    for data in datas:
        j = 0
        for ans in data[1]:
            qstresult[i,:,:,0] = sts2vec3(data[0],max_sequence_len)
            ansresult[i,:,:,0] = sts2vec3(ans,max_sequence_len)
            if isinstance(data[2][j],int):
                tofresult[i,0] = float(data[2][j])
            else:
                tofresult[i,0] = float(int(data[2][j].strip('\ufeff')))
            j += 1
            i += 1
    return (qstresult,ansresult,tofresult)





def data2vec4(data,max_sequence_len=200):
    length = 2 * len(data[1]) - 2
    qstresult = np.zeros([length,max_sequence_len,400,1],'float32')
    ansresult = np.zeros([length,max_sequence_len,400,1],'float32')
    tofresult = np.zeros([length,1],'float32')
    arraynum = 0
    for ansnum,ans in enumerate(data[1]):
        qstresult[arraynum,:,:,0] = sts2vec3(data[0],max_sequence_len)
        ansresult[arraynum,:,:,0] = sts2vec3(ans,max_sequence_len)
        if isinstance(data[2][ansnum],int):
            tofresult[arraynum,0] = float(data[2][ansnum])
        else:
            tofresult[arraynum,0] = float(int(data[2][ansnum].strip('\ufeff')))  
        arraynum += 1
        if tofresult[arraynum,0] == 1:
            for addi in range(len(data[1])-2):
                qstresult[arraynum,:,:,0] = qstresult[arraynum-addi-1,:,:,0]
                ansresult[arraynum,:,:,0] = ansresult[arraynum-addi-1,:,:,0]
                tofresult[arraynum,0] = 1
                arraynum += 1
    return (qstresult,ansresult,tofresult)

def datas2vec10(datas,max_sequence_len=200):
    '''
    输出为（问题，正确回答，错误回答）
    '''
    length = 0
    for data in datas:
        length += len(data[1]) - 1
    qstresult = np.zeros([length,max_sequence_len,400,1],'float32')
    tansresult = np.zeros([length,max_sequence_len,400,1],'float32')
    fansresult = np.zeros([length,max_sequence_len,400,1],'float32')
    arraynum = 0
    for datanum,data in enumerate(datas):
        tofarray = np.array([data[2]],'float32').T
        try:
            tnum = np.where(tofarray==1)[0][0]
        except:
            return(np.array([]),np.array([]),np.array([]))
        for ansnum,ans in enumerate(data[1]):
            if tnum==ansnum:continue
            qstresult[arraynum,:,:,0] = sts2vec3(data[0],max_sequence_len)
            tansresult[arraynum,:,:,0] = sts2vec3(data[1][tnum],max_sequence_len)
            fansresult[arraynum,:,:,0] = sts2vec3(ans,max_sequence_len)
            arraynum += 1 
    return (qstresult,tansresult,fansresult)



def yieldData2(datas,nums=1):
    '''
    对应于上头的datas2vec10
    '''
    while True:
        data = getRandomData(datas,nums)
        qstresult,tansresult,fansresult = datas2vec10(data,200)
        yield (qstresult,tansresult,fansresult)
        
def yieldData3(datas,nums=1):
    '''
    对应于上头的datas2vec10
    '''
    while True:
        data = getRandomData(datas,nums)
        qstresult,tansresult,fansresult = datas2vec10(data,200)
        n = qstresult.shape[0]
        r = np.ones([n,1],'float32')
        if not r.shape[0]:
            continue        
        yield ([np.concatenate((qstresult,qstresult)),np.concatenate((tansresult,fansresult))],
                np.concatenate((r,np.zeros([n,1],'float32'))))   

def randomVecs(qst,ans,res):
    length = len(qst)
    newrange = [ i for i in range(length)]
    random.shuffle(newrange)
    nqst,nans,nres = np.zeros(qst.shape,'float32'),np.zeros(ans.shape,'float32'),np.zeros(res.shape,'float32')
    for i in range(length):
        nqst[i,:,:,:] = qst[newrange[i],:,:,:]
        nans[i,:,:,:] = ans[newrange[i],:,:,:]
        nres[i,0] = res[newrange[i],0]
    return (nqst,nans,nres)


def yieldData4(datas,nums=1):
    '''
    random
    '''
    while True:
        data = getRandomData(datas,nums)
        qstresult,tansresult,fansresult = datas2vec10(data,200)
        length = len(qstresult)
        n = qstresult.shape[0]
        r = np.ones([n,1],'float32')
        if not r.shape[0]:
            continue        
        qst = np.concatenate((qstresult,qstresult),0)
        ans = np.concatenate((tansresult,fansresult),0)
        res = np.concatenate((r,np.zeros([n,1],'float32')),0)
        qst,ans,res = randomVecs(qst,ans,res)
        yield([qst,ans],res)







def getRandomData(datas,randomnums=200):
    return [ datas[i] for i in sorted(random.sample(range(len(datas)), randomnums))]

def yieldData(datas,nums=1):
    while True:
        data = getRandomData(datas,nums)
        qstresult,ansresult,tofresult = datas2vec3(data,200)
        yield ([qstresult,ansresult],tofresult)
        

SAVEVECSDIR = './vecs'
def saveVecs(datas, div = 100):
    i = 0
    maxlength=len(datas)
    while True:
        if div*i>= maxlength:break
        tempmax = maxlength if (div*i+div-1) >= maxlength else div*i+div-1
        tempdata = datas[div*i : tempmax] 
        qstresult,ansresult,tofresult = datas2vec3(tempdata,200)
        np.save(os.path.join(SAVEVECSDIR,'./qst',('%d.npy'%i)),qstresult)
        np.save(os.path.join(SAVEVECSDIR,'./ans',('%d.npy'%i)),ansresult)
        np.save(os.path.join(SAVEVECSDIR,'./tof',('%d.npy'%i)),tofresult)
        print('[%d] write %d ~ %d .     [end] '%(i,div*i,tempmax))
        i += 1
        

#test datalenth


        
        
        
        
        
        
        
        