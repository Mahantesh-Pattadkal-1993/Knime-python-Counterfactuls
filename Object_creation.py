# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:18:19 2021

@author: mpatt
"""


import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression


#Reading the data
data = pd.read_csv("C:\\Users\\mpatt\\knime-workspace-AP4.2\\Counterfactual analysis\\Counterfactual_View_Python_testcase\\adult_2500.csv")
data.head()



#import the custom class

import os
os.chdir("C:\\Users\\mpatt\\knime-workspace-AP4.2\\Counterfactual analysis\\Counterfactual_View_Python_testcase")

from Custom_Class_definition import data_processing


data_train = data.head(2000)

data_test = data.tail(400)

# provide Target column and Positive class
inst = data_processing(data_train, "income", "<=50K")

# Preprocess the training data 
data_train_norm, y_train = inst.pre_process(data_train,norm=True,class_label=True)

data_test_norm, y_test = inst.pre_process(data_test,norm=True,class_label=True)

#y = inst.y


#n = inst.feature_names()

#data_train_df = pd.DataFrame(data_train_norm)
#data_train_df.columns = n

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(data_train_norm, y_train)


from sklearn import svm
from sklearn.gaussian_process.kernels import RBF
#Create a svm Classifier
clf = svm.SVC(probability=True,kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(data_train_norm, y_train)


from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(data_train_norm, y_train)



from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(data_train_norm, y_train)




from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
QDA =  QuadraticDiscriminantAnalysis()
QDA.fit(data_train_norm, y_train)



#Evaluate the model
result = logreg.score(data_test_norm, y_test)
print("Accuracy: %.3f%%" % (result*100.0))


import pickle

os.chdir("C:\\Users\\mpatt\\knime-workspace-AP4.2\\Counterfactual analysis\\Counterfactual_View_Python_testcase")
filename = 'model.pkl'
pickle.dump(logreg, open(filename, 'wb'))
#pickle.dump(logreg, open(filename, 'wb'))



import pickle
filename = 'preprocess.pkl'
pickle.dump(inst, open(filename, 'wb'))





