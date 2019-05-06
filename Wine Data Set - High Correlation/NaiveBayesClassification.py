### Imports

import pandas as pd
import copy as cp
import numpy as np

### Functions

def Read_CSV_as_DataFrame():
    Data = pd.read_csv("Wine.csv")
    return Data

def Split_Data_to_Train_and_Test(Data):
    DataLen = len(Data)
    TestLen = 0.1*DataLen # Train:Test = 80:20
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

def GaussianProbabilityDensity(x,Mean,StdDev):
    Exp = np.exp(-1*(((x-Mean)*(x-Mean))/(2*StdDev*StdDev)))
    return Exp*(1/(np.sqrt(2*np.pi*StdDev*StdDev)))

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

ClassDividing = []

for i in range(NumOfOutcomes):
    Train = [False]*LenOfTrain
    for j in range(LenOfTrain):
        if(TrainSet.iloc[j][Attributes[ColumnNumOfOutcome]] == AllOutcomes[i]):
            Train[j] = True
    ClassDividing.append(Train)

ClassDataFrames = []

for i in range(NumOfOutcomes):
    DF = TrainSet[ClassDividing[i]]
    ClassDataFrames.append(DF)

# Convert DataFrame to List of Lists using df.values.tolist() where df is DataFrame

ListOfList_ClassDataFrames = []

for i in range(NumOfOutcomes):
    DF = ClassDataFrames[i].values.tolist()
    ListOfList_ClassDataFrames.append(DF)

MeanOfClass = []
StdDevOfClass = []

for i in range(NumOfOutcomes):
    Mean = [0]*NumOfAttributes
    StdDev = [0]*NumOfAttributes
    MeanOfClass.append(Mean)
    StdDevOfClass.append(StdDev)

for i in range(NumOfAttributes):
    for j in range(NumOfOutcomes):
        MeanOfClass[j][i] = np.mean(ClassDataFrames[j][Attributes[i]])
        StdDevOfClass[j][i] = np.std(ClassDataFrames[j][Attributes[i]])

# Predicting

PredictedRes = [-1]*LenOfTest

ListOfList_TestSet = TestSet.values.tolist()

for i in range(LenOfTest):
    Test = ListOfList_TestSet[i]
    Probability = [1]*NumOfOutcomes
    for j in range(NumOfAttributes):
        for k in range(NumOfOutcomes):
            if(StdDevOfClass[k][j]==0.0):
                StdDevOfClass[k][j]=0.01
            Probability[k] *= GaussianProbabilityDensity(Test[j],MeanOfClass[k][j],StdDevOfClass[k][j])
    PredictedRes[i] = AllOutcomes[np.argmax(Probability)]        

# Accuracy

Y = list(TestSet[Attributes[ColumnNumOfOutcome]])

CorrectPred = (np.array(PredictedRes)==np.array(Y)).tolist().count(True)
WrongPred = (np.array(PredictedRes)==np.array(Y)).tolist().count(False)

print("% of Accuracy",CorrectPred*100/(CorrectPred+WrongPred))

