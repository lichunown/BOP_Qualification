#encoding:utf8
import os,codecs
import numpy as np
import jieba
import random
READMODEL = False

#我使用的spyder，会保存上一次脚本运行后的数据。
#下面只是检测model是否曾经读入过。没有读入遍置READMODEL为真，重新读model
#这个model是分词模型。Wiki语料真的很大。。真的很大。。


#哦，对了，官方给的数据有一个很奇怪的字符'\ufeff'。需要手动去除。否则脚本会出问题
try:
    model
except: 
    READMODEL = True

dataDir = './BoP2017_DBAQ_dev_train_data' # 数据目录
trainFilename = 'BoP2017-DBQA.train.txt' #训练数据集名字
devFilename = 'BoP2017-DBQA.dev.txt' # 开发数据集名字
testFilename = 'BoP2017-DBQA.test.txt'# 测试数据集名字
decode = 'utf8' # 数据集编码

WORDVECNUM = 400 #和之前词典模型有关，看你word2vec怎么训练的

IGNORE = {'(',')','.','\,','/','\\','?','<','>','\'','[',']','"','-',
          '_','%','#','@','!','$','^','&','*','+','=',
          '。','，','！','《','》','“','”','‘','’','：','；','【',
          '】','、','——','…',''
    }# jieba分词后忽略的词


IGNOREWARNING = True # 是否屏蔽输出分词溢出警告

if READMODEL:# 没有预先读入model，就读model
    import gensim
    model = gensim.models.Word2Vec.load("wiki.zh.text.model")
    
    


"""
数据存储:
    [
     [quetion,ans,result],
     [quetion,ans,result],
     ...
    ]

"""

def readDataFromTxt(filename,decode='utf8'):# 从开发集和训练集中读取数据
    f = codecs.open(filename,'r',decode)
    datas = []
    for line in f:
        datas.append([string.replace('\r\n','').replace('\ufeff','') for string in line.split('\t')])
    return datas

        
testData = readDataFromTxt(os.path.join(dataDir,testFilename)) #测试集数据
trainData = readDataFromTxt(os.path.join(dataDir,trainFilename))# 训练集数据
devData = readDataFromTxt(os.path.join(dataDir,devFilename))# 开发集数据


def sts2list(sts):#jieba 分词，返回分词列表
    return [item for item in jieba.cut(sts) if item not in IGNORE]


# 这个函数我没有用到过，但你可以用啊
def sts2vec_add0(sts,max_sequence_len=200):# max_sequence_len：词的长度。return m*max_sequence_len 矩阵，词不在字典则添加0向量
    sts = sts2list(sts)
    vec = np.zeros([max_sequence_len,WORDVECNUM],'float32')    
    i = 0
    for word in sts:
        try:
            vec[i,:] = model[word]
        except KeyError:
            vec[i,:] = np.zeros(WORDVECNUM,'float32')
        i += 1
        if i >= max_sequence_len:
            if not IGNOREWARNING:
                print('[warning] sequence too long:%d'%len(sts))
            break
            #raise KeyError('sequence too long:%d'%len(sts))
    return vec

def sts2vec(sts,max_sequence_len=200):# max_sequence_len：词的长度。return m*max_sequence_len 矩阵，词不在字典则忽略
    sts = sts2list(sts)
    vec = np.zeros([max_sequence_len,WORDVECNUM],'float32')    
    i = 0
    for word in sts:
        try:
            vec[i,:] = model[word]
            i += 1
        except KeyError:
            pass      
        if i >= max_sequence_len:
            if not IGNOREWARNING:
                print('[warning] sequence too long:%d'%len(sts))
            break
            #raise KeyError('sequence too long:%d'%len(sts))
    return vec

def datas2vec(datas,max_sequence_len=200):# 转换data为向量，返回([qst_vec,ans_vec],result)
    length = len(datas)
    qstresult = np.zeros([length,max_sequence_len,WORDVECNUM],'float32')
    ansresult = np.zeros([length,max_sequence_len,WORDVECNUM],'float32')
    tofresult = np.zeros([length,2],'float32')
    for tempi,data in enumerate(datas):
        qstresult[tempi,:,:] = sts2vec(data[1],max_sequence_len)
        ansresult[tempi,:,:] = sts2vec(data[2],max_sequence_len)
        r = int(data[0])
        if r: tofresult[tempi,:] = np.array([1,0])
        else: tofresult[tempi,:] = np.array([0,1])
    return ([qstresult,ansresult],tofresult)

def datas2vec_Test(datas,max_sequence_len=200):
    length = len(datas)
    qstresult = np.zeros([length,max_sequence_len,WORDVECNUM],'float32')
    ansresult = np.zeros([length,max_sequence_len,WORDVECNUM],'float32')
    for tempi,data in enumerate(datas):
        qstresult[tempi,:,:] = sts2vec(data[0],max_sequence_len)
        ansresult[tempi,:,:] = sts2vec(data[1],max_sequence_len)
    return [qstresult,ansresult]
        
def yieldData(datas,nums=20,weight = []):
    def out(datas,weight):
        x,result = datas2vec(datas)
        if not result.shape[0]:
            print(x,result)
            return None
        if not weight:
            return (x,result)   
        else:
            tn = np.where(result[:,0]==1)[0] 
            fn = np.where(result[:,0]==0)[0] 
            rweight = np.zeros([result.shape[0]],'float32')
            rweight[tn] = weight[0]
            rweight[fn] = weight[1]
            return (x,result,rweight)
    length = len(datas)
    while True:
        for i in range(int(length/nums)):
            yield out(datas[i*nums:i*nums+nums],weight)
        yield out(datas[int(length/nums)*nums:len(datas)],weight)



def yieldTestData(datas = testData, nums = 20):
    def out(datas):
        x,result = datas2vec_Test(datas)
        if not result.shape[0]:
            return None
        return (x,result)          
    length = len(datas)
    while True:
        for i in range(int(length/nums)):
            yield out(datas[i*nums:i*nums+nums])
        yield out(datas[int(length/nums)*nums:len(datas)])


def trueOfDataPercent(datas):
    truenum = 0
    for data in datas:
        if int(data[0]):
            truenum += 1
    return truenum/len(datas)


# 不推荐保存。。。
SAVEVECSDIR = './vecs'
def saveVecs(datas, div = 100):
    i = 0
    maxlength=len(datas)
    while True:
        if div*i>= maxlength:break
        tempmax = maxlength if (div*i+div-1) >= maxlength else div*i+div-1
        tempdata = datas[div*i : tempmax] 
        qstresult,ansresult,tofresult = datas2vec(tempdata,200)
        np.save(os.path.join(SAVEVECSDIR,'./qst',('%d.npy'%i)),qstresult)
        np.save(os.path.join(SAVEVECSDIR,'./ans',('%d.npy'%i)),ansresult)
        np.save(os.path.join(SAVEVECSDIR,'./tof',('%d.npy'%i)),tofresult)
        print('[%d] write %d ~ %d .     [end] '%(i,div*i,tempmax))
        i += 1
        

#test datalenth




        
        
        
        
        
        
        
        