# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 12:43:03 2019

@author: Carla Pastor

Python 3.6.5 |Anaconda, Inc.| IPython 6.4.0 -- An enhanced Interactive Python.

Practice with python tools. Libraries used : numpy, pandas, matplotlib, seaborn, and sklearn.

Business Problem: Breast Cancer Classification using Machine Learning. 

Note: The steps listed mention a general scheme of work.

"""
# After understand & define the objectives for the problem... (Questions pending = 0)
  
''' STEP 1: Data Acquisition '''
# Importing the Data from the online source provided

# First import the libraries 
import pandas as pd      # for data manipulation using dataframes
import numpy as np       # for data statistical analysis 
import matplotlib.pyplot as plt # for data visualisation
import seaborn as sns    # for statistical data visualization

# Import the data from the Sklearn library
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer() # created an instance 

cancer # check all 


''' STEP 2: Data Preparation (or Data cleaning and data transformation)
Understanding the data. '''

cancer.keys() # check my keys 
# output: dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])  

print(cancer['data'])

print(cancer['target']) # 0 or 1 

print(cancer['DESCR'])

print(cancer['target_names']) 
#output: ['malignant' 'benign']

print(cancer['feature_names']) 

cancer['data'].shape


# At this point the data imported did not need preparation, since it is not expected to find problems like
# inconsistent datatype, misspelled attributes, missing or duplicate values, etc. or modificy it data struct. 
    
    
''' STEP 3: Exploratory Data Analysis '''
# Visualization of the features variables to see what can I do with the data.

# pandas data frame 
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target'])) 

df_cancer.head()

df_cancer.tail()

sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )

sns.countplot(df_cancer['target'], label = "Count") 

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)

# Correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
plt.figure(figsize=(20,10)) 

sns.heatmap(df_cancer.corr(), annot=True) 

''' Found: Input: 30 Features. Target class: 2. Class Distribution: 212 Malignant, 357 Benign.
    Dateset : Number of instances: 569 '''
    
    
''' Now, at this point my object is to find the best line that separate the 2 classes observed. '''

''' STEP 4: Data Modeling '''

''' The model that will be used is the "Supported Vector Machine Classifier" to find that best line (Hyperplane) we can use support vectors,
 and the maximun margin between them. In two dimentional space this hyperplane is a line dividing a plane in two parts
 where in each class lay in either side.
The SVM is a discriminative classifier, its algorithm outputs an optimal hyperplane which categorizes new examples. 
'''
# So, adding the target label coloumns

X = df_cancer.drop(['target'],axis=1)

X # check it (569 rows × 30 columns)

y = df_cancer['target']

y # check it (Name: target, Length: 569)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)

X_train.shape

X_test.shape

y_train.shape

y_test.shape

from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, y_train)


''' STEP 5: Evaluation, Visualization, and Comunication'''

y_predict = svc_model.predict(X_test) 
# y_predict    # - to test the output of my model (so far) 

cm = confusion_matrix(y_test, y_predict)  # apply Confussion matrix to visualize the performance of the algorithm, shows the correct and misclassified 

sns.heatmap(cm, annot=True)
# cm; 0, 48, 0, 66 ...terrible, so, we need to work on it.

print(classification_report(y_test, y_predict))

# Improvement, training and testing again is needed to avoid problems like have an overfitted model, i.e. that only work w/ my trainning data 
# my goal here is to get a Generalized model that work w/ all data.

'''Improvement: 1) Data Normalization, and 2) Parameters Optimization'''

'''1) Data Normalization. C parameters (Regularization) to tells the SVM optimization 
how much you want to avoid misclassifying each training example. '''

min_train = X_train.min()

range_train = (X_train - min_train).max()

X_train_scaled = (X_train - min_train)/range_train

sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)

sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)

#normalization in the training dataset
min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


'''Training the model using the normalized (scaled) dataset '''

from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)

''' Re-Evaluation'''

y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm,annot=True,fmt="d")
# output: 43, 5 ,0, 66. a little better

'''2) Gamma parameters, define how far the influence of a single training example reaches, 
 low values meaning ‘far’ and high values meaning ‘close’. '''
 
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
 
from sklearn.model_selection import GridSearchCV
 
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)

grid.fit(X_train_scaled,y_train)

grid.best_params_   # to choose the best values that I have

grid.best_estimator_  # choose the best values that I have

grid_predictions = grid.predict(X_test_scaled)

cm = confusion_matrix(y_test, grid_predictions) # compute my cm again

sns.heatmap(cm, annot=True)
#output: 45, 3, 0, 66 ... this means 0 Type II errors and only 3 Type I errors.

print(classification_report(y_test,grid_predictions)) # Report 


 
''' STEP 5: Deploy and Maintains the model¶'''

# N/A 
























