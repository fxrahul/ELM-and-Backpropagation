#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:50:57 2020

@author: rahul
"""
#-------------------------------------------------Importing libraries-------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import chain
#--------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Initializing Parameters--------------------------------------------------
bs = 0.1
learningRate = 0.01
noOfClasses = 2
noOfHiddenLayers = 4
#--------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Defining Functions------------------------------------------------------

#--------------------------------------------------Sigmoid Activation------------------------------------------------------
def sig(x):
    calculate = 1/(1+np.exp(-x))
    return calculate
#--------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Relu Activation---------------------------------------------------------
def relu(x):
    return max(0,x)
#---------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Softmax Activation-------------------------------------------------------  
def softmax(x):
    exp = np.exp(x)
    calculate = exp / exp.sum()
    return calculate
#--------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Accuracy Calculation-----------------------------------------------------
def predAccuracy(originalLabel,predicted):
        matched = 0
        for i in range(len(originalLabel)):
                if originalLabel[i] == predicted[i]:
                    matched += 1
        accuracyVal = matched / float(len(originalLabel))      
        return accuracyVal
#--------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Local Gradient Calculation----------------------------------------------
def localGradient(s):
    calculate = s*(1-s)
    return calculate
#--------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------Hidden Layer Calculation-----------------------------------------------
def neuronsOperation(row,neurons,we):
    
    finalOutputs = []
    
    if len(we) == 0:
        weights = []
    else:
        weights = np.array(we).reshape(neurons,len(row)) 
        
    for noOfNode in range(neurons): 
        output = 0
        if len(we) == 0:
            weight = np.random.rand(len(row))
            weights.append(weight)
        else:
            weight = weights[noOfNode] 

        for i in range(len(row)):
#            print("weight: ",weight)
            output += row[i]*weight[i]
        finalOutputs.append( sig(output) + bs)

    return finalOutputs,list( chain.from_iterable(weights) )
#--------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Testing Function--------------------------------------------------------
def test(inputTest, outputTest, allFinalWeights):
    predictedOutput = []
    m = 0
    for row in inputTest:
            
            inputs = row
            calculatedOutput = []      
            neurons = 4
            calculatedOutput.append(row)
            
            for k in range(noOfHiddenLayers+1):
                if k == noOfHiddenLayers:
                    neurons = noOfClasses 
                    
                inputs,w = neuronsOperation(inputs,neurons,allFinalWeights[m][k])
                calculatedOutput.append(inputs)
            predictedOutput.append( np.argmax( softmax( calculatedOutput[-1]  ) ) )

            m = m+1
    print("Testing Accuracy: ",predAccuracy(outputTest,predictedOutput) * 100 , "%" )
#--------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Training Function--------------------------------------------------------
def train(inputTrain,outputTrain):
    allFinalWeights =[]
    for epoch in range(23):
        
        
        allHiddenLayersOutput = []
        predictedOutput = []
        m = 0
        for row in inputTrain:
            
            
            inputs = row
            
            calculatedOutput = []
            finalWeights =[]
            neurons = 4
            calculatedOutput.append(row)
            for k in range(noOfHiddenLayers+1):
                
                if k == noOfHiddenLayers:
                    neurons = noOfClasses 
                if epoch == 0:
                    inputs,w = neuronsOperation(inputs,neurons,[])
                else:
                    inputs,w = neuronsOperation(inputs,neurons,allFinalWeights[m][k])
                    
                finalWeights.append(w)         
                calculatedOutput.append(inputs)

            predictedOutput.append( np.argmax( softmax( calculatedOutput[-1]  ) ) )
            allHiddenLayersOutput.append(calculatedOutput)
            
            if epoch == 0:            
                allFinalWeights.append(finalWeights)
        
            m = m+1

        print("Training Epoch : ", epoch+1,"............." )

#------------------------------------------Backpropagation----------------------------------------------------------------  
        for q in range(len(allHiddenLayersOutput)-1,-1,-1):
            
    
            for r in range(len(allFinalWeights[q])-1,-1,-1):
            
                
                
                noOfPreviousLayerNeurons = len(allHiddenLayersOutput[q][r])
                l = 0
                e = 0
                for t in range(len(allFinalWeights[q][r])-1,-1,-1):      
                    if l > noOfPreviousLayerNeurons-1:
                        l = 0
                        e = e + 1
                #-----------------------------------------Updating Weights--------------------------------------------
                    allFinalWeights[q][r][t] = allFinalWeights[q][r][t] - learningRate * localGradient(allHiddenLayersOutput[q][r+1][e]) * allHiddenLayersOutput[q][r][l]
                #-----------------------------------------------------------------------------------------------------    
                    l = l + 1
#--------------------------------------------------------------------------------------------------------------------------
                    
    return allFinalWeights
#--------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------            

#------------------------------------------------Main Function-------------------------------------------------------------  
if __name__ == "__main__":
    data = pd.read_csv("transfusion.csv")
    wholeDataset = pd.DataFrame(data)
    wholeDataset = wholeDataset.astype(float)
    inputData = wholeDataset.drop(columns=[wholeDataset.columns[-1]]).to_numpy()
    outputLabel = wholeDataset[wholeDataset.columns[-1]].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(inputData, outputLabel, test_size=0.33,random_state = 42)
    y_train = y_train.reshape(len(y_train),1)
    y_test = y_test.reshape(len(y_test),1)
    
#------------------------------------------------Training Start------------------------------------------------------------
    wt = train(X_train,y_train) 
#------------------------------------------------Training End---------------------------------------------------------------   

#------------------------------------------------Testing Start--------------------------------------------------------------
    test(X_test,y_test,wt)
#--------------------------------------------Testing End------------------------------------------------------------------ 

    