### Imports

import pandas as pd
import copy as cp
import numpy as np

import warnings
warnings.filterwarnings("ignore")

### Functions

def Read_CSV_as_DataFrame():
    Data = pd.read_csv("CarDataSet.csv")
    return Data

def Split_Data_to_Train_and_Test(Data):
    DataLen = len(Data)
    TestLen = 0.2*DataLen # Train:Test = 80:20
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


### Code

Data = Read_CSV_as_DataFrame()

Attributes = list(Data.columns) # Attributes are same order as DataFrame

TrainANDTest = Split_Data_to_Train_and_Test(Data)
TrainSet = TrainANDTest[0]
TestSet = TrainANDTest[1]
LenOfTrain = len(TrainSet)
LenOfTest = len(TestSet)

NumOfAttributes = len(Attributes)-1 # Removing Class Value

# Assuming OutCome is last Attribute

ColumnNumOfOutcome = NumOfAttributes
AllOutcomes = list(set(Data[Attributes[ColumnNumOfOutcome]]))
NumOfOutcomes = len(AllOutcomes)

Y_TestSet = list(TestSet[Attributes[NumOfAttributes]])

Y_TrainSet = np.matrix(TrainSet[Attributes[NumOfAttributes]]).T

del TrainSet[Attributes[NumOfAttributes]]

# Bottom added x0 = 1

X_TestSet = np.array(TestSet.values.tolist())

X_TestSet = np.append(X_TestSet,np.array([[1]]*LenOfTest),axis=1)

X_TrainSet = np.matrix(TrainSet.values.tolist())

X_TrainSet = np.append(X_TrainSet,np.array([[1]]*LenOfTrain),axis=1)

theta = np.matrix([1]*(NumOfAttributes+1)).T

# Gradient Descent applying 500 times with alpha = 0.01

for k in range(500):
    h = X_TrainSet*theta
    h = -h
    h = 1/(1+np.exp(h))
    for j in range(NumOfAttributes+1):
        temp = np.matrix([0]*(NumOfAttributes+1)).T
        for i in range(LenOfTrain):
            temp[j] = temp[j]+(h[i]-Y_TrainSet[i])*X_TrainSet[i,j]
        theta = theta - 0.01*temp    

# Prediction

PredictedRes = [0]*LenOfTest

for i in range(LenOfTest):
    Temp = 0
    for j in range(NumOfAttributes+1):
        Temp = Temp+(theta[j]*X_TestSet[i,j])
    if(Temp>=0):
        PredictedRes[i] = 1

# Accuracy

CorrectPred = (np.array(PredictedRes)==np.array(Y_TestSet)).tolist().count(True)
WrongPred = (np.array(PredictedRes)==np.array(Y_TestSet)).tolist().count(False)

print("% of Accuracy",CorrectPred*100/(CorrectPred+WrongPred))
