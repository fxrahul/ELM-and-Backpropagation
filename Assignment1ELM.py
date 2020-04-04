#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:21:41 2020

@author: rahul
"""
#--------------------------------------------------Importing Libraries----------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import chain
#-------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------Intializing Parameters------------------------------------------------
bs = 0.1
noOfNeurons = 6
#-------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------Defining Functions----------------------------------------------------

#---------------------------------------------------Calculation Output---------------------------------------------------
def predValue(inputs,w):
    output = sig( np.dot(inputs, w) + bs )
    return output
#-------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------Sigmoid Activation----------------------------------------------------
def sig(x):
    calculate = 1/(1+np.exp(-x))
    return calculate
#-------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Softmax Activation-----------------------------------------------------
def softmax(x):
    exp = np.exp(x)
    calculate = exp / exp.sum()
    return calculate
#-------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------Accuracy Prediction---------------------------------------------------
def predAccuracy(originalLabel,predicted):

        matched = 0
        for i in range(len(originalLabel)):
                if originalLabel[i] == predicted[i]:
                    matched += 1
        accuracyVal = matched / float(len(originalLabel))      
        return accuracyVal
#--------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------Training Function-----------------------------------------------------
def train(train_input_data,outputTrainLabel):

    hMatrix = []
    finalWeights = []
    for row in train_input_data:
        h =[]
        weights = []
        for i in range(noOfNeurons):
            output = 0
            weight = np.random.rand(len(row))
            weights.append(weight)
            for j in range(len(row)):
                output += row[j]*weight[j]
            h.append(sig(output) + bs)
        hMatrix.append(h)
        finalWeights.append(weights)
        
    beta = np.dot(np.linalg.pinv(hMatrix), outputTrainLabel)

    return beta,finalWeights
#--------------------------------------------------------------------------------------------------------------------------
    
#-----------------------------------------------------Testing Function-----------------------------------------------------
def test(data, outputD,b,weights):
    hMatrix = []
    m = 0
    for row in data:
        weight = weights[m]
        h =[]
        for i in range(noOfNeurons):
            output = 0
            we = weight[i]
            for j in range(len(row)):
                output += row[j]*we[j]
            h.append(sig(output) + bs)
        hMatrix.append(h) 
        m += 1
 
    o = np.dot(hMatrix , b)
    
    o[ o >= 0.5 ] = 1
    o[ o < 0.5 ] = 0
    
    acc = predAccuracy(outputD, o)
    print("Testing Accuracy",acc*100,"%")
#--------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------
    
 #----------------------------------------------------Main Function-------------------------------------------------------   
if __name__ == "__main__":
    data = pd.read_csv("transfusion.csv")
    wholeDataset = pd.DataFrame(data)
    wholeDataset = (wholeDataset).astype(float)
    inputData = wholeDataset.drop(columns=[wholeDataset.columns[-1]]).to_numpy()
    outputLabel = wholeDataset[wholeDataset.columns[-1]].to_numpy()

   #---------------------------------------------------Train Test Splitting------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(inputData, outputLabel, test_size=0.33,random_state = 42)
   #-----------------------------------------------------------------------------------------------------------------------
    
    y_train = y_train.reshape(len(y_train),1)
    y_test = y_test.reshape(len(y_test),1)
    #---------------------------------------------------Training Start-----------------------------------------------------
    beta,weights = train(X_train,y_train) 
    #---------------------------------------------------Training End-------------------------------------------------------
    
    #---------------------------------------------------Testing Start------------------------------------------------------
    test(X_test,y_test,beta,weights)
    #---------------------------------------------------Testing End---------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------
    
    
    