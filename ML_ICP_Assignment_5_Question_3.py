#!/usr/bin/env python
# coding: utf-8

# In[4]:


# importing necessary libraires for executing the code

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder


# In[5]:


df = pd.read_csv("Iris.csv") # imprting the iris csv file into df variable
df.head() # printing the top 5 rows of the iris dataframe


# In[9]:


stdsc = StandardScaler() # Calling the StandardScaler method to the stdsc variable
X_train_std = stdsc.fit_transform(df.iloc[:,range(0,4)].values) # fitting the input values to the X_train_std variable

class_le = LabelEncoder() # Calling the method LabelEncoder() to the class_le varaible 
y = class_le.fit_transform(df['Species'].values) # Giving the output predictable data to the variable y
lda = LinearDiscriminantAnalysis(n_components=2) # Calling the Linear Discriminant Analysis with 2 components to lda variable
X_train_lda = lda.fit_transform(X_train_std,y) # Training the model with X_train as input and y as output variables
data=pd.DataFrame(X_train_lda) # Taking the input data as a dataframe  
data['class']=y # appending the output data to the input dataframe 
data.columns=["LD1","LD2","class"] # Renaming the col names
data.head()  # printing the top 5 rows of the data


# In[8]:


# Giving the necessary information to plot the data 
markers = ['s', 'x', 'o']
colors = ['r', 'b', 'g']
sns.lmplot(x="LD1", y="LD2", data=data, hue='class', markers=markers, fit_reg=False, legend=False)
plt.legend(loc='upper center')
plt.show() # printing the plot


# In[ ]:


# Question 4
# PCA is an unsupervised learning method whereas LDA is a supervised learning method.
#PCA is used for feature extraction and data compression, LDA is used for feature selection and classification.

