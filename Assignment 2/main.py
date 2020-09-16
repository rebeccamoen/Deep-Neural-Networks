#Importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

#Reading the Data File
data = pd.read_csv('ecoli.csv')
data = shuffle(data)

print(data['SITE'].unique())
print("Number of rows and columns:", data.shape)

#Deleting the column "SEQUENCE_NAME"
data = data.drop(columns="NAME")

data=data.replace(to_replace="cp",value="0")
data=data.replace(to_replace="im",value="1")

# split into input (X) and output (Y) variables
splitratio = 0.8

X_train = data.iloc[:int(len(data)*splitratio),0:6]
X_val = data.iloc[:int(len(data)*splitratio),0:6]
Y_train = data.iloc[:int(len(data)*splitratio),7]
Y_val = data.iloc[:int(len(data)*splitratio),7]

print(X_train)
print(Y_train)

#Import MLP classifier model from sklearn
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=1000, activation='relu')

#Fit the data into the model
mlp.fit(X_train,Y_train)

#From scratch
weights = [-0.1, 0.20, -0.23, -0.1, 0.20, -0.23, -0.1, 0.20, -0.23]

import math

def sigmoid(z):
    if(z<-100):
        return 0
    if(z>100):
        return 1
    return 1.0/math.exp(-z)

def firstLayer(row,weights):
    activation_1 = weights[0]*1
    activation_1 += weights[1]*row[0]
    activation_1 += weights[2]*row[1]

    activation_2 = weights[3]*1
    activation_2 += weights[4]*row[2]
    activation_2 += weights[5]*row[3]
    return sigmoid(activation_1),sigmoid(activation_2)

def secondLayer(row,weights):
    activation_3 = weights[6]
    activation_3 += weights[7]*row[0]
    activation_3 += weights[8]*row[1]
    return sigmoid(activation_3)

def predict(row,weights):
    input_layer = row
    first_layer = firstLayer(input_layer,weights)
    second_layer = secondLayer(first_layer,weights)
    return second_layer,first_layer

for d in data:
    print(predict(d,weights)[0],d[-1])   #Prints y_hat and y

def train_weights(train,learningrate,epochs):
    for epoch in range(epochs):
        sum_error = 0.0
        for row in train:
            prediction,first_layer = predict(row,weights)
            error = row[-1]-prediction
            sum_error += error
            #First layer
            weights[0] = weights[0] + learningrate*error*1
            weights[3] = weights[3] + learningrate*error

            weights[1] = weights[1] + learningrate*error*row[0]
            weights[2] = weights[2] + learningrate*error*row[1]
            weights[4] = weights[4] + learningrate*error*row[2]
            weights[5] = weights[5] + learningrate*error*row[3]

            #Second layer
            weights[6] = weights[6] + learningrate*error
            weights[7] = weights[7] + learningrate*error*first_layer[0]
            weights[8] = weights[8] + learningrate*error*first_layer[1]
        if((epoch%100==0) or (last_error != sum_error)):
            print("Epoch "+str(epoch) + " Learning rate " + str(learningrate) + " Error " + str(sum_error))
        last_error = sum_error
    return weights

learningrate = 0.0001 #0.00001
epochs = 1000
train_weights = train_weights(data,learningrate,epochs)
print(train_weights)

#Predicting the values
#y_pred = mlp.predict(X_test)

#Printing Accuracy, Classification Report and the Confusion Matrix
#print("Accuracy : {}%".format(accuracy_score(y_test, y_pred)*100))
#print("Classification Report: \n", classification_report(y_test, y_pred))
#print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
