### Imports

import pandas as pd
import copy as cp
import numpy as np
import math
import operator

### Functions

def Read_CSV_as_DataFrame():
    Data = pd.read_csv("CarDataSet.csv")
    return Data

def Split_Data_to_Train_and_Test(Data):
    DataLen = len(Data)
    TestLen = 0.3*DataLen # Train:Test = 70:30
    Test = [False]*DataLen
    Train = [True]*DataLen
    Temp = 0
    while(Temp < TestLen):
        Index = np.random.randint(DataLen)
        if(Test[Index] == False):
            Test[Index] = True
            Train[Index] = False
            Temp = Temp + 1
    TestSet = Data[Test]
    TrainSet = Data[Train]
    return [TrainSet,TestSet]

def GetNeighbors(TrainingSet,TestPoint,K):
    a = len(TestPoint)
    Distances = []
    for i in range(len(TrainingSet)):
        dist = 0
        for j in range(a):
            dist=dist+(TrainingSet[i][j]-TestPoint[j])**2
        Distances.append((i,math.sqrt(dist)))
    Distances.sort(key=operator.itemgetter(1))
    Neighbors = []
    for i in range(K):
        Neighbors.append(Distances[i][0])
    return Neighbors

Data = Read_CSV_as_DataFrame()

Attributes = list(Data.columns) # Attributes are same order as DataFrame

NumOfAttributes = len(Attributes)-1 # Removing Class Value

ColumnNumOfOutcome = NumOfAttributes

AllOutcomes = list(set(Data[Attributes[ColumnNumOfOutcome]]))

TrainANDTest = Split_Data_to_Train_and_Test(Data)
TrainSet = TrainANDTest[0]
TestSet = TrainANDTest[1]

ListOfTestSet = np.array(TestSet.values.tolist())

ListOfTrainSet = np.array(TrainSet.values.tolist())

Predicted=[-1]*len(ListOfTestSet)
for i in range(len(ListOfTestSet)):
    Neighbors = GetNeighbors(ListOfTrainSet,ListOfTestSet[0],50)
    A = [0]*len(AllOutcomes)
    for j in range(50):
        Temp = ListOfTrainSet[Neighbors[j]][len(ListOfTrainSet[0])-1]
        A[int(Temp)-1]+=1
    Predicted[i] = np.argmax(A)

Count = 0
for i in range(len(ListOfTestSet)):
    if(Predicted[i]+1==ListOfTestSet[i][len(ListOfTestSet[0])-1]):
        Count+=1
print("Accuracy is ",100*Count/len(ListOfTestSet),"%")
