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
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics


# In[7]:


df= pd.read_csv("CC GENERAL.csv")  # Taking the input file and storing it as a dataframe df
df.head()  # Printing the top 5 rows of the dataframe 


# In[9]:


print(df['TENURE'].value_counts())  # Printing the frequency of the Tenure Column in the dataframe
x = df.iloc[:,[1,2,3,4]] # Taking the 4 columns as X data
y = df.iloc[:,-1] # taking the Tenure Column as Y
print(x) # Printing the X
print(y) # Printing the variable Y


# In[10]:


le = preprocessing.LabelEncoder() # # # Calling the StandardScaler method to the stdsc variable
df['CUST_ID'] = le.fit_transform(df.CUST_ID.values) # Fitting the Cust Id 
pca2 = PCA(n_components=2) # Calling PCA module with 2 components
principalComponents = pca2.fit_transform(x) # Fitting the X into PCA
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2']) #Creating a dataframe with 3 columns for variable x
finalDf = pd.concat([principalDf, df[['TENURE']]], axis = 1)# Adding the variable y to the dataframe
finalDf.head() # printing the top 5 rows of the dataframe 


# In[20]:


# Applying Kmeans to pca data
x_pca = finalDf[['principal component 1','principal component 2']] # Taking  'principal component 1', 'principal component 2' as X
y_pca = finalDf[['TENURE']]  # Taking the Tenure as Y
nclusters = 2 # this is the k in kmeans 
km = KMeans(n_clusters=nclusters) # Calling Kmeans clustring to the variable m
km.fit(x_pca) # Fitting the train data into the model

y_cluster_kmeans_pca = km.predict(x_pca) # predicting the test data accuracy 

score_pca = metrics.silhouette_score(x_pca, y_cluster_kmeans_pca) # calculating the accuracy of the model output
print(score_pca) # printing the output score


# In[11]:


# Applyinh means to raw data
nclusters = 2 # this is the k in kmeans 
km = KMeans(n_clusters=nclusters) # Calling Kmeans clustring to the variable m
km.fit(x)  # Fitting the train data into the model

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x) # predicting the test data accuracy 

score = metrics.silhouette_score(x, y_cluster_kmeans)   # calculating the accuracy of the model output
print(score) # printing the output score


# In[12]:


# Applying Scaling to raw data
scaler = StandardScaler()  # Taking the class as Y
X_Scale = scaler.fit_transform(x) # Fitting the X data to Scaler method.

pca2 = PCA(n_components=2) # Calling PCA with 2 components
principalComponents = pca2.fit_transform(X_Scale) # Fitting the X into PCA

principalDf1 = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])  #Creating a dataframe with 2 columns for variable x

finalDf1 = pd.concat([principalDf1, df[['TENURE']]], axis = 1) # Adding the variable y to the dataframe
print(finalDf1.head()) # printing the top 5 rows of the dataframe 


# In[21]:


# Applying Kmeans to scaled pca data
x_pca_scaled = finalDf1[['principal component 1','principal component 2']] # Taking  'principal component 1', 'principal component 2' as X of scaled data
y_pca_scaled = finalDf1[['TENURE']] # # Taking the Tenure as Y from scaled data
nclusters = 2 # this is the k in kmeans 
km = KMeans(n_clusters=nclusters) # Calling Kmeans clustring to the variable m for scaled data
km.fit(x_pca_scaled)  # Fitting the train scaled data into the model

y_cluster_kmeans_pca_scaled = km.predict(x_pca_scaled) # # predicting the test data accuracy  for scaled data

score_pca_Scaled = metrics.silhouette_score(x_pca_scaled, y_cluster_kmeans_pca_scaled)    # calculating the accuracy of the model output
print(score_pca_Scaled) # printing the output score 


# In[13]:


# applying model for scaled raw data
nclusters = 2 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)  # Calling Kmeans clustring to the variable m for scaled raw data
km.fit(X_Scale)   # Fitting the train scaled raw data into the model

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_Scale) # # predicting the test data accuracy  for scaled raw data
score = metrics.silhouette_score(X_Scale, y_cluster_kmeans) # calculating the accuracy of the model output for scaled raw data
print(score) # Printig the output score for scaled raw data

