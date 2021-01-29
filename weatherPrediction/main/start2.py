#!/usr/bin/env python
# coding: utf-8

# # read data and clean data

# In[1]:


import pandas as pd 
import numpy as np 

trainTable = pd.read_csv("train.csv", engine = "python", encoding = "big5")

trainTable.head(18)


# In[2]:


trainTable.shape


# In[3]:


trainDataTable = trainTable["0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23".split(" ")]
trainDataTable = trainDataTable.replace("NR", "0").astype("float")


# In[4]:


trainIndex1 = (trainDataTable.shape[0] // 18 // 3) * 18
trainIndex2 = (trainDataTable.shape[0] // 18 // 3) * 2 * 18

train1 = trainDataTable.iloc[0:trainIndex1, :]
train2 = trainDataTable.iloc[trainIndex1:trainIndex2, :]
testSet = trainDataTable.iloc[trainIndex2:, :]


# In[5]:


print(train1.shape)
print(train2.shape)
print(testSet.shape)


# In[6]:


# this is for mean model 
def readData(df, rowStep, columnStep):
    xList = []
    yList = []
    shape = df.shape
    columns = shape[1]
    rows = shape[0]
    for i in range(0, rows, rowStep):
        for j in range(0, columns - columnStep):  # looping for 15 column
            dataset = df.iloc[i: i + rowStep, j:j + 9]
            dataset.astype("float32")
            pm = df.iloc[i + 9, j + 9]
            npArr = np.array(dataset).astype("float32")
            avgArr = npArr.mean(axis = 1)
            avgArr = avgArr.reshape(18,1)
            xList.append(avgArr)
            yList.append(float(pm))
    
    return xList, yList


# In[7]:



realMeanModelX1, realMeanModelY1 = readData(train1, 18, 9)
realMeanModelX2, realMeanModelY2 = readData(train2, 18, 9)
realMeanModelXAll, realMeanModelYAll = readData(pd.concat([train1, train2]), 18, 9)


# In[20]:


realMeanModelYAll[0:5]


# In[9]:


realMeanModelXAll[0:3]


# In[10]:


# find the mean for all data 
# find the standard deviation 

def getMean(df, arr):
    output = []
    for label in arr:
        print(label)
        labelDf = df.loc[:,label,:]
        avg = np.array(labelDf, dtype = "float").mean()
        print("avg = " + str(avg))
        output.append(avg)
    return output

def getDV(df, arr):
    output = []
    for label in arr:
        print(label)
        labelDf = df.loc[:,label,:]
        avg = np.array(labelDf, dtype = "float").std()
        print("avg = " + str(avg))
        output.append(avg)
    return output


# In[11]:


trainTable.replace("NR", "0", inplace = True)
indexTable = trainTable.set_index(["日期", "測項"])
indexDropTable = indexTable.drop(["測站"], axis=1)  

meanArr = getMean(indexDropTable, ["CH4", "CO", "NO", "NO2", "PM2.5", "RAINFALL"])
stdArr = getDV(indexDropTable, ["CH4", "CO", "NO", "NO2", "PM2.5", "RAINFALL"])


# In[12]:


def getstdDistribution(df, indexArr, meanArr, stdArr):
    output = []
    index = 0
    for i in indexArr:
        row = df.iloc[i,:]
        row = (row - meanArr[index]) / stdArr[index]
        output.append(row)
        index = index + 1
    return output
    


# In[13]:


# this is for model 2 
# CH4, CO, NO, NO2, PM 2.5, Rain fall

def readData2(df, rowStep, columnStep, meanArr, stdArr):
    xList = []
    yList = []
    shape = df.shape
    columns = shape[1]
    rows = shape[0]
    for i in range(0, rows, rowStep):
        for j in range(0, columns - columnStep):  # looping for 15 column
            dataset = df.iloc[i: i + rowStep, j:j + 9]
            pm = df.iloc[i + 9, j + 9]
            arr = getstdDistribution(dataset, [1,2,4,5,9,10], meanArr, stdArr)
            npArr = np.array(arr)
            xList.append(npArr.reshape(npArr.shape[0] * npArr.shape[1], 1))
            yList.append(float(pm))
    
    return xList, yList


# In[14]:


def getPower(x):
    outputList = []
    for i in range(len(x)):
        arr = x[i]
        arr.reshape((18,1))
        power2 = np.power(arr, 2)
        power3 = np.power(arr, 3)
        arr = np.vstack((arr, power2, power3))
        outputList.append(arr)
    
    return outputList


# In[16]:


realModelX1, realModelY1 = readData2(train1, 18, 9, meanArr, stdArr)
realModelX2, realModelY2 = readData2(train2, 18, 9, meanArr, stdArr)
realModelXAll, realModelYAll = readData2(pd.concat([train1, train2]), 18, 9, meanArr, stdArr)


# In[18]:


print(len(realModelXAll))   # check data
print(len(realModelYAll))
print(realModelX2[1199])
print(realModelY2[1199])


# # finish cleaning and getting data 

# # start inital training

# In[21]:


learningRate = 0.9
regulationRate = 0.00001
epoch = 1000


# loss = (y - y1)^2 + sum(wi^2)
# y1 = w1 * x1 + w2 * x1^2 + w3 * x1^3 + w4 * x2 + w5 * x2^2 ..... + b

# in other way for wn
# loss = (cn * wn + c2)^2 + regRate * wn^2 + c3

# so
# dloss/dw = 2(cn * wn + c2) * cn + 2regRate * wn

# for b 
# loss = (c + b)^2 + c1
# dloss/db = 2[c + b]

def sgdm(realX, realY, learningRate, regRate, epoch, experience):
    w = np.random.randn(1, len(realX[0]))
    b = np.random.rand(1,1)
    print("w = ")
    print(w, "\n")
    print("b = ")
    print(b, "\n")
    
    mw = 0  # momanton
    mb = 0
    for i in range(epoch):
        errorW = np.zeros((1, w.shape[1]))
        errorB = 0
        for j in range(len(realY)): # looping number of dataset
            errorW = errorW + (2 * (np.dot(w, realX[j]) + b - realY[j]) * realX[j].T) + w * regRate
            errorB = errorB + (2 * (np.dot(w, realX[j]) + b - realY[j]))
            
        avgErrorB = errorB / len(realY)
        avgErrorW = errorW / len(realY)
            
#         print loss
        if i % 50 == 0:
            loss = (np.dot(w, realX[j]) + b - realY[j])**2
            print("loss = " + str(loss))  
            
#         updating
        mb = experience * mb + (1 - experience) * avgErrorB    
        b = b - mb * learningRate
        mw = experience * mw + (1 - experience) * avgErrorW   
        w = w - mw * learningRate
        
    return w, b
    


# # start training

# In[22]:


learningRate = 0.000001
regulationRate = 0.00001
epoch = 1000

# loss = (y - y1)^2 + sum(wi^2)
# y1 = w1 * x1 + w2 * x1^2 + w3 * x1^3 + w4 * x2 + w5 * x2^2 ..... + b

# in other way for wn
# loss = (cn * wn + c2)^2 + regRate * wn^2 + c3

# so
# dloss/dw = 2(cn * wn + c2) * cn + 2regRate * wn

# for b 
# loss = (c + b)^2 + c1
# dloss/db = 2[c + b]

def train2(realX, realY, learningRate, regRate, epoch):
    w = np.random.randn(1, len(realX[0]))
    b = np.random.rand(1,1)
    print("w = ")
    print(w, "\n")
    print("b = ")
    print(b, "\n")
    for i in range(epoch):
        errorW = np.zeros((1, w.shape[1]))
        errorB = 0
        for j in range(len(realY)): # looping number of dataset
            errorW = errorW + (2 * (np.dot(w, realX[j]) + b - realY[j]) * realX[j].T) + w * regRate
            errorB = errorB + (2 * (np.dot(w, realX[j]) + b - realY[j]))
            
        avgErrorB = errorB / len(realY)
        avgErrorW = errorW / len(realY)
            
#         print loss
        if i % 50 == 0:
            loss = (np.dot(w, realX[j]) + b - realY[j])**2
            print("loss = " + str(loss))  
            
#         updating
        b = b - avgErrorB * learningRate
        w = w - avgErrorW * learningRate
        
    return w, b
    


# In[28]:


train2(realMeanModelXAll, realMeanModelYAll, 0.0000001, 0.00001, 1000)


# In[29]:


wM1, bM1= sgdm(realMeanModelXAll, realMeanModelYAll, 0.000005, 0.00001, 3000, 0.9)


# In[45]:


w1, b1 = sgdm(realModelXAll, realModelYAll, 0.00001, 0.00001, 1000, 0.99)


# In[35]:


def getAvgError(w, b, dataX, dataY):
    error = 0
    for i in range(len(dataY)):
        y = (np.dot(w, dataX[i]) + b)
        error = error + abs(dataY[i] - y)
    return error / len(dataY)


# In[36]:


getAvgError(wM1, bM1, realMeanModelX1, realMeanModelY1)


# In[37]:


getAvgError(wM1, bM1, realMeanModelX2, realMeanModelY2)


# In[46]:


getAvgError(w1, b1, realModelX1, realModelY1)


# In[52]:


getAvgError(w1, b1, realModelX2, realModelY2)


# In[49]:


testMeanX, testMeanY = readData(testSet, 18, 9)
testRealX, testRealY = readData2(testSet, 18, 9, meanArr, stdArr)


# In[50]:


getAvgError(wM1, bM1, testMeanX, testMeanY)


# In[51]:


getAvgError(w1, b1, testRealX, testRealY)


# mean model seen like look better so output mean model

# In[53]:


wdf = pd.DataFrame(wM1)
bdf = pd.DataFrame(bM1)
wdf.to_csv("weights.csv")
bdf.to_csv("bias.csv")


# In[ ]:




