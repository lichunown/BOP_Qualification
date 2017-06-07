# 2017BOP QA问答系统

## 简介

这是我们在BOP资格赛搞的一个小小的QA问答系统。感谢lsm，scs，nl等人的帮助。

参加比赛的代码实在是太简陋了，各种solution，我在想要不要重构一番。。。

所以……先等我重构完吧。

**2017/6/7 [update]**
哇，花了一晚上时间，算是重构了一遍吧。主要是改了数据处理的那部分。嗯，感觉存储速度有了一定上升吧。。。

注释写了一点，改天我把剩下的注释写完？（“改天”大概是无期限喽。。）

## 使用
### 安装

**python3.5环境。**
**建议使用linux系统。 **

工具推荐使用anaconda。
这个脚本使用的jieba分词，genism的word2vec工具。
机器学习使用的tensorflow后端，keras前端。

环境配置，大概也就
```bash
pip install jieba
pip install gensim
pip install tensorflow  #GPU和CPU版本不同，对系统和python环境也有要求。。
pip install keras
```

### 训练分词模型
我是参考的[中英文维基百科语料上的Word2Vec实验](http://www.52nlp.cn/%E4%B8%AD%E8%8B%B1%E6%96%87%E7%BB%B4%E5%9F%BA%E7%99%BE%E7%A7%91%E8%AF%AD%E6%96%99%E4%B8%8A%E7%9A%84word2vec%E5%AE%9E%E9%AA%8C)。

#### 处理wiki数据集

采用中文维基语料训练模型。
首先，下载中文维基数据集https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2。

然后，执行脚本
```bash
python process_wiki.py zhwiki-latest-pages-articles.xml.bz2 wiki.zh.text
```
#### 简体繁体处理
使用opencc，把语料的繁体中文转换为简体中文。具体安装配置请参考[Open Chinese Convert 開放中文轉換](https://github.com/BYVoid/OpenCC#installation-安裝)
**注意，新版本的配置脚本改了，和之前我提到的blog说的不一样。**
```
opencc -i wiki.zh.text -o wiki.zh.text.jian -c t2s.json
```
#### word2vec训练
```
python train_word2vec_model.py wiki.zh.text.jian.seg wiki.zh.text.model wiki.zh.text.vector
```

### 玄学模型
有了分好的词向量，就可以用玄学（机器学习）的方法对数据进行训练了。

我相信我的小学生版本的脚本应该很好读懂的。。。（好吧，注释什么的慢慢补充）

**哦，对了，官方给的数据有一个很奇怪的字符'\ufeff'。需要手动去除。否则脚本会出问题**

## 脚本介绍
### readData.py
主要是用来数据处理的……
### train_LSTM_CNN.py
用的是LSTM-CNN模型，至于为什么是这个模型，我也不知道。。。反正是copy加魔改版的。。。
### CreatePredictResult.py & CreatePredictResult_dev.py
用来生成最终提交数据的。。
### other

不用管其他的了，那只是我瞎几把写的，啥乱用没有。。。

哦不，还是有点用处的，至少比全随机的MRR值高。。。
