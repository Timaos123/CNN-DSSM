# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # CNN

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
                seqList,
                preModelPath="chinese_L-12_H-768_A-12",
                learning_rate=0.1,
                hiddenSize=None
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
        # print(y_true, y_pred)
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

import pickle as pkl
with open("model/CNN_org_syn_initList.pkl","rb") as CNN_org_syn_initListFile:
    CNN_org_syn_initList=pkl.load(CNN_org_syn_initListFile)
myCNNSyn=MyCNN(CNN_org_syn_initList)

# %%
myCNNSyn.model.load_weights("model/CNN_org_syn")

with open("model/CNN_org_syn.pkl","rb") as myModelFile:
    hpDict=pkl.load(myModelFile)
myCNNSyn.learning_rate=hpDict["learning_rate"]
myCNNSyn.maxLen=hpDict["maxLen"]
myCNNSyn.myOHEncoder=hpDict["myOHEncoder"]
myCNNSyn.preModelPath=hpDict["preModelPath"]
myCNNSyn.tokenizer=hpDict["tokenizer"]
myCNNSyn.vocabSize=hpDict["vocabSize"]

# %% [markdown]
# # BM25

# %%
import pkuseg
from gensim.summarization import bm25

class BM25():
    def __init__(self, opList, user_dict=None):
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

    def cal_BM25_sim(self, sentence: str):
        sentence=sentence.lower()
        query = self.seg.cut(sentence)
        # print(query)
        scores = self.bm25Model.get_scores(query)
        tmp = list(zip(self.instructions,scores))
        index_and_score = sorted(tmp, key=lambda x: x[1], reverse=True)
        index_and_score=[row for row in index_and_score if row[1]>0]
        index_and_score=pd.DataFrame(index_and_score).drop_duplicates().values.tolist()
        return index_and_score
        
with open("./data/orgSynSampleList.pkl","rb") as orgSynSampleListFile:
    orgSynSampleList=pkl.load(orgSynSampleListFile)
orgSynSampleList=[[rowItem.replace(" ","").strip().lower() if type(rowItem)==str else rowItem for rowItem in row] for row in orgSynSampleList]
dtForBM25=[row for row in orgSynSampleList if row[2]==1]
myBM25=BM25(dtForBM25)
# %% [markdown]
# # combine

# %%
import numpy as np
import pickle as pkl
from nltk.util import ngrams

with open("model/tfidf_org_syn.pkl","rb") as tfidfModelFile:
    tfidfModel=pkl.load(tfidfModelFile)

def ngramCutSent(sentence):
    return " ".join(["".join(ngItem) for ngItem in ngrams([""]+[cItem for cItem in sentence]+[""],3)])

def getKNgram(mySent):
    sV=tfidfModel.transform([ngramCutSent(mySent)])
    svList=sV.todense().tolist()[0]

    iwDict=dict(list(zip(tfidfModel.vocabulary_.values(),tfidfModel.vocabulary_.keys())))
    wpList=[(iwDict[i],svList[i]) for i in range(len(iwDict))]

    kwList=[row[0] for row in list(sorted(wpList,key=lambda wp:wp[1],reverse=True)) if row[1]>0.1]
    return kwList


# %%
orgList=CNN_org_syn_initList
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
    preYC,preYP=myCNNSyn.predict([tryOrgList,synNameList])

    tpList=preYP[:,1].tolist()
    opList=list(zip(tryOrgList,tpList))
    opList=sorted(opList,key=lambda row:row[1],reverse=True)
    outputList=[opItem for opItem in opList[:5] if opItem[1]>0.5]
    return outputList

with open("model/orgSuffix.pkl","rb") as orgSuffixFile:
    newEndList=pkl.load(orgSuffixFile)
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
if __name__=="__main__":
    inputStr="宣传部门"
    print("输入内容：",inputStr)
    print("输出内容：",getSyn(inputStr)[0][0])


