# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:50:42 2021

@author: mpatt
"""

#Importing the libraries
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
#std_scale = preprocessing.StandardScaler()


class data_processing():
    
  def __init__(self, input_data, target_column, positive_class):
    #super(Learner,self).__init__(copy=True, with_mean=True, with_std=True)
    self.input_data = input_data
    self.temp_data = input_data.copy()
    self.target_column = target_column 
    self.positive_class = positive_class
    self.X = pd.DataFrame()
    self.Y = pd.DataFrame()
    self.output = pd.DataFrame()
    self.data_numerical = pd.DataFrame()
    self.std_scale = preprocessing.StandardScaler()

  def target_binarization(self):
    
    """
        The Given target column will be categorical, but the scikit-learn requires it to be a binary vector
        So, this function converts eg: male , female to 1,0 based on the positive class specified by the user

        input_data     =   Provided datatype in pandas dataframe
        target_column  =   Name of the target column in string format
        positive_class =   Name of the positive class

    """
    
    self.temp_data["Target_Class"] = np.where(self.temp_data[self.target_column] ==self.positive_class,1,0)
    self.y = self.temp_data["Target_Class"]
    self.X = self.temp_data.drop([self.target_column,"Target_Class"], axis=1)
      
    return self.X, self.y


  def normalisation(self):
    self.data_numerical = self.temp_data.select_dtypes(exclude=['object'])
    self.data_numerical = self.data_numerical.drop('Target_Class', 1)
    self.std_scale.fit(self.data_numerical)

  
  def pre_process(self,external_data,norm=True,class_label=False):
    print("hello")
    
    self.target_binarization()
    self.normalisation()
    
    """
    X          = Given input data should be provided as a pandas dataframe
    norm       = if  True, outputs z-transform of the input table, else outputs inverse z-transform 
    output     = normalised or denormalised data in numpy array format
    
    The z-transform are calculated based on the parameters of the training data
    
    """
    if (class_label):
        
        external_data["Target_Class"] = np.where(external_data[self.target_column] ==self.positive_class,1,0)
        y_label = external_data["Target_Class"] 
        external_data= external_data.drop("Target_Class", 1)
        print(external_data.columns)
        if(norm):
            external_data = external_data.select_dtypes(exclude=['object'])
            
           
            #print(data.head)
            self.output = self.std_scale.transform(external_data)
        else:
            external_data = external_data.select_dtypes(exclude=['object'])
           
            self.output = self.std_scale.inverse_transform(external_data)    
        
        
        
    else:
        
        if(norm):
            external_data = external_data.select_dtypes(exclude=['object'])
            #print(data.head)
            self.output = self.std_scale.transform(external_data)
            y_label = None
        else:
            external_data = external_data.select_dtypes(exclude=['object'])
            self.output = self.std_scale.inverse_transform(external_data)
            y_label = None
        
    return self.output , y_label


  def feature_names(self):
    """
    Outputs the names of the numerical columns in order as given in the input data
    
    """
    return self.data_numerical.columns


      