# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Loading Data and Preprocessing

# %%
import pickle as pkl

#读取数据
with open("./data/orgSynSampleList.pkl","rb") as orgSynSampleListFile:
    orgSynSampleList=pkl.load(orgSynSampleListFile)

#数据预处理
orgSynSampleList=[[rowItem.replace(" ","").strip().lower() if type(rowItem)==str else rowItem for rowItem in row] for row in orgSynSampleList]
# orgSynSampleList=[row[1].replace(" ","").strip().lower() for row in orgSynSampleList]


#构建组织列表
orgList=list(set([row[0] for row in orgSynSampleList]))

#构建(尾缀,尾缀数量)列表
endList=[row[-1] for row in orgList]
endItemList=list(set(endList))
enList=[]
for endItemItem in endItemList:
    enList.append((endItemItem,endList.count(endItemItem)))


#构建基于尾缀数量进行排序的列表
enList=list(sorted(enList,key=lambda row:row[1],reverse=True))


#存储尾缀列表
newEndList=[row[0] for row in enList[:12]]
with open("model/orgSuffix.pkl","wb+") as orgSuffixFile:
    pkl.dump(newEndList,orgSuffixFile)
# %% [markdown]
# # Tfidf-cosine Similarity

# %%
#依照n-gram进行句子切分的函数（经验上看Tri-gram效果最好）
def ngramCutSent(sentence):
    return " ".join(["".join(ngItem) for ngItem in ngrams([""]+[cItem for cItem in sentence]+[""],3)])

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams

#构建模型
TOrgList=[ngramCutSent(row) for row in orgList]
tfidfModel=TfidfVectorizer()
tfidfModel.fit(TOrgList)


#构造基于tfidf筛选出重要n-gram的函数
import numpy as np
def getKNgram(mySent):
    sV=tfidfModel.transform([ngramCutSent(mySent)])
    svList=sV.todense().tolist()[0]

    iwDict=dict(list(zip(tfidfModel.vocabulary_.values(),tfidfModel.vocabulary_.keys())))
    wpList=[(iwDict[i],svList[i]) for i in range(len(iwDict))]

    kwList=[row[0] for row in list(sorted(wpList,key=lambda wp:wp[1],reverse=True)) if row[1]>0.1]
    return kwList


#存储tfidf模型
import pickle as pkl
with open("model/tfidf_org_syn.pkl","wb+") as tfidfModelFile:
    pkl.dump(tfidfModel,tfidfModelFile)

# %% [markdown]
# # CNN Similarity

# %%
import numpy as np
import jieba
import pandas as pd
import os
import tqdm
import bert
from tensorflow import keras
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
from sklearn.preprocessing import OneHotEncoder
import pickle as pkl
from sklearn.metrics import f1_score


# %%
class MyCNN:

    def __init__(self,
                seqList,#组织列表（用于提取组织名的最大长以及词汇）
                preModelPath="chinese_L-12_H-768_A-12",#bert预训练模型path，主要需要相关的config以及词汇
                learning_rate=0.1,#学习率
                hiddenSize=None#隐藏层大小，默认为240
            ):
        self.preModelPath=preModelPath
        self.learning_rate=learning_rate
        self.hiddenSize=hiddenSize
        self.buildVocab(seqList)
        self.tokenizer=bert.bert_tokenization.FullTokenizer(os.path.join(self.preModelPath, "overVocab.txt"), do_lower_case=True)
        self.buildModel()

    def buildVocab(self,seqList):
        vocabList=[]
        self.maxLen=0
        for row in seqList:
            if len(row)>self.maxLen:
                self.maxLen=len(row)
            for token in row:
                vocabList.append(token)
        print("max length:{}".format(self.maxLen))
        vocabList=list(set(vocabList))
        with open(os.path.join(self.preModelPath,"vocab.txt"),"r",encoding="utf8") as vocabFile:
            oriVocabList=[row.strip() for row in tqdm.tqdm(vocabFile)]
        vocabList=oriVocabList+vocabList
        self.vocabSize=len(vocabList)
        with open(os.path.join(self.preModelPath,"bert_config.json"),"r",encoding="utf8") as confFile:
            confJson=json.load(confFile)
            confJson["vocab_size"]=self.vocabSize
            if self.hiddenSize is None:
                self.hiddenSize=confJson["hidden_size"]
        # print(confJson)
        with open(os.path.join(self.preModelPath,"bert_config.json"),"w",encoding="utf8") as confFile:
            json.dump(confJson,confFile,indent=2)
        print("vocab size:{}".format(self.vocabSize))
        with open(os.path.join(self.preModelPath,"overVocab.txt"),"w+",encoding="utf8") as overVocabFile:
            for row in vocabList:
                overVocabFile.write(row+"\n")

    def amsoftmax_loss(self,y_true, y_pred, scale=30, margin=0.35):
        print(y_true, y_pred)
        y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
        y_pred *= scale
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)

    def getCosLoss(self,y_true, y_pred, scale=30, margin=0.35):
        loss=(2-2*y_true[1])*y_pred
        return loss
    
    def buildModel(self):

        bert_params = bert.params_from_pretrained_ckpt(self.preModelPath)
        
        inputLayer1 = keras.layers.Input(shape=(self.maxLen,), dtype='int32')
        embeddingLayer1 = keras.layers.Embedding(input_dim=self.vocabSize+1,output_dim=self.hiddenSize,input_length=self.maxLen,)(inputLayer1)
        reshapeLayer1=keras.layers.Reshape((self.maxLen,self.hiddenSize,1))(embeddingLayer1)
        
        inputLayer2 = keras.layers.Input(shape=(self.maxLen,), dtype='int32')
        embeddingLayer2 = keras.layers.Embedding(input_dim=self.vocabSize+1,output_dim=self.hiddenSize,input_length=self.maxLen)(inputLayer2)
        reshapeLayer2=keras.layers.Reshape((self.maxLen,self.hiddenSize,1))(embeddingLayer2)

        cnnLayer1=keras.layers.Conv2D(3,kernel_size=(3,self.hiddenSize))(reshapeLayer1)
        poolingLayer1=keras.layers.MaxPool2D(pool_size=(3,1))(cnnLayer1)
        flattenLayer1=keras.layers.Flatten()(poolingLayer1)
        denseLayer1=keras.layers.Dense(128,activation="tanh")(flattenLayer1)
        denseLayer1=keras.layers.Dense(64,activation="tanh")(denseLayer1)
        denseLayer1=keras.layers.Dense(32,activation="tanh")(denseLayer1)

        cnnLayer2=keras.layers.Conv2D(3,kernel_size=(3,self.hiddenSize))(reshapeLayer2)
        poolingLayer2=keras.layers.MaxPool2D(pool_size=(3,1))(cnnLayer2)
        flattenLayer2=keras.layers.Flatten()(poolingLayer2)
        denseLayer2=keras.layers.Dense(128,activation="tanh")(flattenLayer2)
        denseLayer2=keras.layers.Dense(64,activation="tanh")(denseLayer2)
        denseLayer2=keras.layers.Dense(32,activation="tanh")(denseLayer2)

        BLC1=keras.layers.LayerNormalization()(denseLayer1)
        BLC2=keras.layers.LayerNormalization()(denseLayer2)
        multLayer=keras.layers.Dot(axes=1,normalize=True)([BLC1,BLC2])
        nlLayer=1-multLayer
        concatLayer=keras.layers.concatenate([nlLayer,multLayer],axis=-1)

        # denseLayer=keras.layers.Dense(64,activation="tanh")(multLayer)
        # denseLayer=keras.layers.Dense(32,activation="tanh")(denseLayer)
        # denseLayer=keras.layers.Dense(16,activation="tanh")(denseLayer)
        

        # outputLayer=keras.layers.Dense(2,name="classifier",activation="softmax")(denseLayer)

        self.model = keras.models.Model([inputLayer1,inputLayer2],concatLayer)
        self.model.compile(loss=self.amsoftmax_loss,
                            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))


    def fit(self,X,y,epochs=1,batch_size=1024):
        '''
        X:cutted seq
        y:cutted y
        '''
        
        self.myOHEncoder=OneHotEncoder()
        classList=[[row] for row in list(set(y.tolist()))]
        self.myOHEncoder.fit(classList)
        y=self.myOHEncoder.transform(y.reshape([-1,1])).toarray().astype(np.float32)

        X1=X[0]
        X2=X[1]
        X1=np.array([self.tokenizer.convert_tokens_to_ids(row) for row in X1])
        X2=np.array([self.tokenizer.convert_tokens_to_ids(row) for row in X2])

        X1=np.array([row+[0]*(self.maxLen-len(row)) if len(row)<self.maxLen else row[:self.maxLen] for row in X1.tolist()]).astype(np.int32)
        X2=np.array([row+[0]*(self.maxLen-len(row)) if len(row)<self.maxLen else row[:self.maxLen] for row in X2.tolist()]).astype(np.int32)
        
        # print(X1,X2,y)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        self.model.fit([X1,X2],y,epochs=epochs,batch_size=batch_size,callbacks=[callback])
        
    def predict(self,X):
        
        X1=X[0]
        X2=X[1]
        X1=np.array([self.tokenizer.convert_tokens_to_ids(row) for row in X1])
        X1=np.array([row+[0]*(self.maxLen-len(row)) if len(row)<self.maxLen else row[:self.maxLen] for row in X1.tolist()]).astype(np.int32)
        X2=np.array([self.tokenizer.convert_tokens_to_ids(row) for row in X2])
        X2=np.array([row+[0]*(self.maxLen-len(row)) if len(row)<self.maxLen else row[:self.maxLen] for row in X2.tolist()]).astype(np.int32)

        preYP=self.model.predict([X1,X2])
        preYC=np.argmax(preYP,axis=-1)

        return preYC,preYP


# %%
#实例化CNN-DSSM模型
myCNN=MyCNN(orgList,learning_rate=0.0001)


# %%
#输出模型架构
myCNN.model.summary()


# %%
#拆分X和y
orgSynSampleArr=np.array(orgSynSampleList)
X=[orgSynSampleArr[:,0],orgSynSampleArr[:,1]]
X[0]=[row.replace(" ","").strip().lower() for row in X[0]]
X[1]=[row.replace(" ","").strip().lower() for row in X[1]]
y=orgSynSampleArr[:,2].astype(np.int32)

#拆分训练测试集，testSize为测试集占比
testSize=0.3
trainSampleIndexList=(np.random.random_sample(int(len(X[0])*(1-testSize)))*len(X[0])).astype(int).tolist()
trainX1=[X[0][sampleIndexItem] for sampleIndexItem in trainSampleIndexList]
trainX2=[X[1][sampleIndexItem] for sampleIndexItem in trainSampleIndexList]
trainX=[trainX1,trainX2]
trainy=np.array([y[sampleIndexItem] for sampleIndexItem in trainSampleIndexList])

#训练模型
myCNN.fit(trainX,trainy,epochs=500,batch_size=1024)


# %%
#测试模型
print("testing model ...")

testSampleIndexList=(np.random.random_sample(int(len(X[0])*testSize))*len(X[0])).astype(int).tolist()

testX1=[X[0][sampleIndexItem] for sampleIndexItem in testSampleIndexList]
testX2=[X[1][sampleIndexItem] for sampleIndexItem in testSampleIndexList]
testX=[testX1,testX2]
testy=np.array([y[sampleIndexItem] for sampleIndexItem in testSampleIndexList])

preYC,preYP=myCNN.predict(testX)#输出分类及分类概率
# print("testX:",testX[:5])
print("preY:",preYC[:5])
print("testY:",testy[:5])

trainPreYC,trainPreYP=myCNN.predict(trainX)
TtrainCY=trainy
print("train f1:",f1_score(TtrainCY,trainPreYC,average="macro"))
TtestCY=testy
TtestPreCY=preYC
print("test f1:",f1_score(TtestCY,TtestPreCY,average="macro"))
from sklearn.metrics import roc_auc_score
print("auc:",roc_auc_score(TtestCY,TtestPreCY))

#存储模型
myCNN.model.save_weights("model/CNN_org_syn")
hpDict={
    "learning_rate":myCNN.learning_rate,
    "maxLen":myCNN.maxLen,
    "myOHEncoder":myCNN.myOHEncoder,
    "preModelPath":myCNN.preModelPath,
    "tokenizer":myCNN.tokenizer,
    "vocabSize":myCNN.vocabSize,
}
with open("model/CNN_org_syn.pkl","wb+") as myModelFile:
    pkl.dump(hpDict,myModelFile)
with open("model/CNN_org_syn_initList.pkl","wb+") as CNN_org_syn_initListFile:
    pkl.dump(orgList,CNN_org_syn_initListFile)





# %% [markdown]
# # BM25

# %%
import pkuseg
from gensim.summarization import bm25

class BM25():
    def __init__(self, 
                opList,#[原始名称,可能名称]列表
                user_dict=None):#用户自定义词典
        if user_dict is None:
            self.seg = pkuseg.pkuseg(user_dict=user_dict, postag=False)
        else:
            self.seg = pkuseg.pkuseg(postag=False)
        self.instructions=[]
        corpus = []
        for row in opList:
            question = row[0]
            corpusRow=[]
            for word in self.seg.cut(question):
                corpusRow.append(word)
            corpus.append(corpusRow)
            self.instructions.append(question)
        self.bm25Model = bm25.BM25(corpus)
        self.corpus = corpus
        #self.average_idf = sum(map(lambda k: float(self.bm25Model.idf[k]), self.bm25Model.idf.keys())) / len(self.bm25Model.idf.keys())

    def cal_BM25_sim(self, sentence: str):#获取备选名称列表（按照相似程度BM25排序）
        sentence=sentence.lower()
        query = self.seg.cut(sentence)
        # print(query)
        scores = self.bm25Model.get_scores(query)
        tmp = list(zip(self.instructions,scores))
        index_and_score = sorted(tmp, key=lambda x: x[1], reverse=True)
        index_and_score=[row for row in index_and_score if row[1]>0]
        index_and_score=pd.DataFrame(index_and_score).drop_duplicates().values.tolist()
        return index_and_score


# %%
#构造BM25数据集（仅真实数据）
dtForBM25=[row for row in orgSynSampleList if row[2]==1]


#实例化BM25模型
myBM25=BM25(dtForBM25)


#存储BM25模型（jupyter可以存储同时读取pkl但是py文件就不行，有亻老能解决这个问题嘛？）
import pickle as pkl
with open("model/BM25_org_syn.pkl","wb+") as BM25ModelFile:
    pkl.dump(myBM25,BM25ModelFile)

# %% [markdown]
# # predict

# %%
def isSubExist(oriStr,subList):
    tmpStr=[]
    for subStrItem in subList:
        if subStrItem in oriStr:
            tmpStr.append(subStrItem)
    tmpStr="".join(tmpStr)
    # print(tmpStr,oriStr,len(tmpStr)/len(oriStr))
    return len(tmpStr)/len(oriStr)

def getSynWithCNN(synName):
    synName=synName.lower()

    kwList=getKNgram(synName)
    tryOrgList=[row for row in orgList if isSubExist(row,kwList)>=0.8]
    if len(tryOrgList)==0:
        return []
    synNameList=[synName for rowI in range(len(tryOrgList))]
    # print(tryOrgList)
    preYC,preYP=myCNN.predict([tryOrgList,synNameList])

    tpList=preYP[:,1].tolist()
    opList=list(zip(tryOrgList,tpList))
    opList=sorted(opList,key=lambda row:row[1],reverse=True)
    outputList=[opItem for opItem in opList[:5] if opItem[1]>0.5]
    return opList

def getSyn(synName):
    #CNN+bm25
    bm25List=[row[0] for row in myBM25.cal_BM25_sim(synName)]
    cnnList=[row[0] for row in getSynWithCNN(synName)]
    newList=list(set(bm25List+cnnList))
    irDict={}
    if len(cnnList)==0:
        return [(row,rowI) for rowI,row in enumerate(bm25List)]
    if len(bm25List)==0:
        return [(row,rowI) for rowI,row in enumerate(cnnList)]
    for newItem in newList:
        if newItem in bm25List and newItem in cnnList:
            irDict[newItem]=1/(bm25List.index(newItem)+1+cnnList.index(newItem)+1)
    irList=list(sorted([(keyItem,irDict[keyItem]) for keyItem in irDict],key=lambda row:row[1],reverse=True))
    
    #规则（尾缀相同）
    suffix=""
    for endItem in newEndList:
        if synName.endswith(endItem):
            suffix=endItem
            break
    irList=[ir for ir in irList if ir[0].endswith(suffix)]

    return irList


# %%
inputStr="宣传"
print("输入内容：",inputStr)
print("实际实体：",getSyn(inputStr)[0][0])

# %% [markdown]
# # BLEU

# %%
BLEUTestList=[row[:2] for row in orgSynSampleList if row[2]==1]
BLEUtestSampleIndexList=(np.random.random_sample(150)*len(BLEUTestList)).astype(int).tolist()
BLEUTestList=[BLEUTestList[rowI] for rowI in BLEUtestSampleIndexList]


# %%
from nltk.translate.bleu_score import sentence_bleu
import tqdm

testYList=[]
bleuScoreList=[]
for rowI in tqdm.tqdm(range(len(BLEUTestList))):
    refList=[BLEUTestList[rowI][0]]
    canList=getSyn(BLEUTestList[rowI][1])
    if len(canList)>0:
        canList=canList[0][0]
    else:
        canList=""
    bleuScore=sentence_bleu(refList,canList)
    bleuScoreList.append(bleuScore)
print("\nbleu:",np.mean(bleuScoreList))


