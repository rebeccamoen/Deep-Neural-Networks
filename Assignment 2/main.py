#Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#Reading the Data File
data=pd.read_csv('ecoli.csv')
data.head()

#Deleting the column "SEQUENCE_NAME"
del data['SEQUENCE_NAME']

#Splitting the data
X = data.iloc[:,0:6]
print(X.head())
y = data.iloc[:,7]
print(y.head())

#Creating train and test set for model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#Fit the data into the model
clf = KNeighborsClassifier()
clf = clf.fit(X_train,y_train)

#Predicting the values
y_pred = clf.predict(X_test)
y_pred

#Printing Accuracy, Classification Report and the Confusion Matrix
print("Accuracy : {}%".format(accuracy_score(y_test, y_pred)*100))
print("Classification Report: \n",classification_report(y_test, y_pred))
conf_mat2 = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', conf_mat2)

