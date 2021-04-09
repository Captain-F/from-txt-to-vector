# from-txt-to-vector
A python package to convert txts to inputs accepted by deep learning models

## 1. 介绍
* 核心功能是将.txt的文本直接转换为深度序列模型接受的输入
* 支持word2vec模型训练与特征抽取
* 支持BERT特征抽取

## 2. 如何使用
`pip install fromTxtToVec`  
* 在你的工程下放置你需要的处理.txt文本（数量为1），txt内容的格式如下
  * 字与字之间换行，句子与句子之间空行，字与标签之间用制表符`\t`分割  
```
例子：
我  B-PER
爱  O I
南  B-LOC
京  I-LOC

我  B-PER
爱  O
上  B-LOC
海  I-LOC
```
* 如果你想搭建一个word2vec-BiLSTM-CRF模型，那么需要做的就是
```
from fromTxtToVec.to_vector import To_vec

sents, labels = To_vec(mode='w2v', sent_maxlen=100).vector()
```
* 如果你想搭建一个BERT-BiLSTM-CRF模型，那么需要做的就是
```
from fromTxtToVec.to_vector import To_vec

sents, labels = To_vec(mode='bert', sent_maxlen=100).vector()
请输入BERT模型的绝对路径or相对路径...[path]
请输入抽取的粒度：token or cls [token]
```
* 如果你想通过训练word2vec生成Embedding层接受的查找表（weights），那么需要做的就是
```
from fromTxtToVec.to_vector import To_vec
from fromTxtToVec.train_vector import Embedding

#调用w2v_matrix函数，word2vec训练的语料是以.txt中的句子构成
matrix = To_vec(mode='w2v', sent_maxlen=100).w2v_matrix(emb_size=100)

#如果想用自己的大规模语料，可以使用Embedding中的w2v函数
#corpus参数接受的数据形如：[['我', '爱', '南', '京'], ..., ]
matrix = Embedding(emb_size=100).w2v(corpus=corpus)
```
## 3. 应用实例
...
