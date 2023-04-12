#!/usr/bin/env python
# coding: utf-8

# In[4]:


# importing necessary libraries to be used in the code
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC, LinearSVC


# In[8]:


df= pd.read_csv("pd_speech_features.csv") # Taking the input file and storing it as a dataframe df
print(df.head()) # Printing the top 5 rows of the dataframe 
print(df.shape) # printing the shape of the dataframe 
print(df['class'].value_counts()) # Printing the frequency of the class Column in the dataframe


# In[9]:


X = df.drop('class',axis=1).values # Taking the entire dataset except class column as X 
y = df['class'].values# Taking the class column as y
scaler = StandardScaler() # # Calling the StandardScaler method to the stdsc variable
X_Scale = scaler.fit_transform(X) # Fitting the X data to Scaler method.


# In[10]:


pca2 = PCA(n_components=3) # Calling PCA with 3 components
principalComponents = pca2.fit_transform(X_Scale) # Fitting the X into PCA
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3']) #Creating a dataframe with 3 columns for variable x
finalDf = pd.concat([principalDf, df[['class']]], axis = 1) # Adding the variable y to the dataframe
finalDf.head() # printing the top 5 rows of the dataframe 


# In[13]:


x_pca = finalDf[['principal component 1', 'principal component 2', 'principal component 3']] # Taking  'principal component 1', 'principal component 2', 'principal component 3' as X 
y_pca = finalDf[['class']] # Taking the class as Y
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(x_pca,y_pca, test_size=0.3,random_state=0) # Applying Test train split to the data 
svc = SVC(max_iter=1000) # Calling SVC method to iterate over 1000 times
svc.fit(X_train_pca, y_train_pca) # Fitting the data to the model
Y_pred_pca = svc.predict(X_test_pca) # predicting the test data from the model
acc_svc_pca = round(svc.score(X_train_pca,y_train_pca) * 100, 2) # Rounding of the SVC model accuracy foor PCA data
print("svm accuracy of Pca =", acc_svc_pca) # printing the accuracy of the model for pca data


# In[11]:


# fINDING THE accuracy for the raw data
X_train, X_test, y_train, y_test = train_test_split(X_Scale,y, test_size=0.3,random_state=0) # Applying Test train split to the data 
svc = SVC(max_iter=1000)  # Calling SVC method to iterate over 1000 times
svc.fit(X_train, y_train)  # Fitting the data to the model
Y_pred = svc.predict(X_test)   # predicting the test data from the model
acc_svc = round(svc.score(X_train, y_train) * 100, 2)  # Rounding of the SVC model accuracy foor PCA data
print("svm accuracy =", acc_svc) # printing the accuracy of the model for pca data

