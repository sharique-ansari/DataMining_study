### Imports

import pandas as pd
import numpy as np
import copy as cp
import operator
import math

### Functions

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

### Code

Data = pd.read_csv("GlassModified.csv")
Data.set_index('id',inplace=True)
TrainSet = Data.iloc[:100]
TestSet = Data.iloc[100:]

Attributes = list(Data.columns)

NumOfAttributes = len(Attributes)-1 # Removing Class Value

ColumnNumOfOutcome = NumOfAttributes

AllOutcomes = list(set(Data[Attributes[ColumnNumOfOutcome]]))

ListOfTrainSet = np.array(TrainSet.values.tolist())
ListOfTestSet = np.array(TestSet.values.tolist())

Predicted=[-1]*len(ListOfTestSet)
for i in range(len(ListOfTestSet)):
    Neighbors = GetNeighbors(ListOfTrainSet,ListOfTestSet[0],5)
    A = [0]*len(AllOutcomes)
    for j in range(5):
        Temp = ListOfTrainSet[Neighbors[j]][len(ListOfTrainSet[0])-1]
        A[int(Temp)]+=1
    Predicted[i] = np.argmax(A)

Count = 0
for i in range(len(ListOfTestSet)):
    if(Predicted[i]==ListOfTestSet[i][len(ListOfTestSet[0])-1]):
        Count+=1
print("Accuracy is ",100*Count/len(ListOfTestSet),"%")
