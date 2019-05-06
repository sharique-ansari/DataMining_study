### Imports

import numpy as np
import pandas as pd
import copy as cp

### Global Variables

Mean=[]
StdDev=[]
Mean_Si=[]
StdDev_Si=[]
Outcomes=[]

### Functions

def trainNB(FeatureMatrix,Labels):
    global Mean
    global StdDev
    NoOfRows=len(FeatureMatrix)
    NoOfCols=len(FeatureMatrix[0])-1# Removing Class Label
    NoOfLabels=len(Labels)
    ListOfClasses=[]
    for i in range(NoOfLabels):
        Temp=[]
        for j in range(NoOfRows):
            if(FeatureMatrix[j][NoOfCols]==Labels[i]):
                Temp.append(FeatureMatrix[j])
        ListOfClasses.append(Temp)
    #print(ListOfClasses)
    for i in range(NoOfLabels):
        Mean_I = [0]*NoOfCols
        StdDev_I = [0]*NoOfCols
        Mean.append(Mean_I)
        StdDev.append(StdDev_I)
    for i in range(NoOfLabels):
        for j in range(NoOfCols):
            Temp=[]
            for k in range(len(ListOfClasses[i])):
                #print(i,k,j)
                #print(ListOfClasses[i][k][j])
                Temp.append((ListOfClasses[i][k][j]))
            Mean[i][j]=np.mean(Temp)
            StdDev[i][j]=np.std(Temp)

def trainNB_Si(FeatureMatrix,Labels):
    global Mean_Si
    global StdDev_Si
    NoOfRows=len(FeatureMatrix)
    NoOfCols=len(FeatureMatrix[0])-1# Removing Class Label
    NoOfLabels=len(Labels)
    ListOfClasses=[]
    for i in range(NoOfLabels):
        Temp=[]
        for j in range(NoOfRows):
            if(FeatureMatrix[j][NoOfCols]==Labels[i]):
                Temp.append(FeatureMatrix[j])
        ListOfClasses.append(Temp)
    #print(ListOfClasses)
    for i in range(NoOfLabels):
        Mean_I = [0]*NoOfCols
        StdDev_I = [0]*NoOfCols
        Mean_Si.append(Mean_I)
        StdDev_Si.append(StdDev_I)
    for i in range(NoOfLabels):
        for j in range(NoOfCols):
            Temp=[]
            for k in range(len(ListOfClasses[i])):
                #print(ListOfClasses[i][k][j])
                Temp.append((ListOfClasses[i][k][j]))
            Mean_Si[i][j]=np.mean(Temp)
            StdDev_Si[i][j]=np.std(Temp)

def classifyNB(TestPoint):
    global Mean
    global StdDev
    global Outcomes
    Probability = [1]*len(Outcomes)
    for i in range(len(TestPoint)-1):
        for j in range(len(Outcomes)):
            Probability[j] *= (np.exp(-1*(((TestPoint[i]-Mean[j][i])*(TestPoint[i]-Mean[j][i]))/(2*StdDev[j][i]*StdDev[j][i]))))*(1/(np.sqrt(2*np.pi*StdDev[j][i]*StdDev[j][i])))
    return Outcomes[np.argmax(Probability)]

def classifyNB_Si(TestPoint):
    global Mean_Si
    global StdDev_Si
    global Outcomes
    Probability = [1]*len(Outcomes)
    for i in range(len(TestPoint)-1):
        for j in range(len(Outcomes)):
            Probability[j] *= (np.exp(-1*(((TestPoint[i]-Mean_Si[j][i])*(TestPoint[i]-Mean_Si[j][i]))/(2*StdDev_Si[j][i]*StdDev_Si[j][i]))))*(1/(np.sqrt(2*np.pi*StdDev_Si[j][i]*StdDev_Si[j][i])))
    return Outcomes[np.argmax(Probability)]


### Code

Data = pd.read_csv("GlassModified.csv")
Data.set_index('id',inplace=True)

Data_with_Si = cp.deepcopy(Data)

#del Data["'Mg'"]
del Data["'Si'"]
#del Data["'Fe'"]

TrainSet = Data[0:100]
TestSet = Data[100:]

TrainSet_Si = Data_with_Si[0:100]
TestSet_Si = Data_with_Si[100:]

Attributes = list(Data.columns)

Outcomes = list(set(TrainSet["'Type'"]))

D_with_Si = []

for i in range(100):
    Temp = list(TrainSet_Si.iloc[i])
    D_with_Si.append(Temp)

D = []

for i in range(100):
    Temp = list(TrainSet.iloc[i])
    D.append(Temp)

trainNB_Si(D_with_Si,Outcomes)
trainNB(D,Outcomes)

T_with_Si = []

for i in range(46):
    Temp = list(TestSet_Si.iloc[i])
    T_with_Si.append(Temp)

T = []

for i in range(46):
    Temp = list(TestSet.iloc[i])
    T.append(Temp)

T_Predicted=[]

for i in range(46):
    T_Predicted.append(classifyNB(T[i]))

T_with_Si_Predicted=[]

for i in range(46):
    T_with_Si_Predicted.append(classifyNB_Si(T_with_Si[i]))

Accuracy=[0]*2

Count = 0
for i in range(46):
    if(T_with_Si_Predicted[i]==T_with_Si[i][len(T_with_Si[0])-1]):
        Count+=1
Accuracy[0]=100*Count/(46)

Count = 0
for i in range(46):
    if(T_Predicted[i]==T[i][len(T[0])-1]):
        Count+=1
Accuracy[1]=100*Count/(46)

print(Accuracy)
