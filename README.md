# In_class_programming_assingment-5
In_class_programming_assingment-5


Question 1:

Apply PCA on CC dataset.

•	We have imported necessary libraries and import the CC dataset csv file to code.
•	We have taken the ‘BALANCE’, ‘BALANCE_FREQUENCY’, ‘PURCHASES’, ‘ONEOFF_PURCHASES’ as X variables and tenure as Y variable. 
•	We applied principal component analysis with 2 components for the given data and stored the data in the dataframe.
•	We have applied Kmeans Clustering with 2 clusters to the above data and we have calculated the accuracy of the model.
•	We are applying the Kmeans clustering to the raw data without PCA and calculating the Accuracy of the model.
•	Now we are applying scaling to the raw data and finding the PCA for the scaled data and then storing the data into a dataframe.
•	Now we are calculating the Kmeans Clustering model accuracy for the scaled pca applied data.
•	Now we are applying the Kmeans clustering to the scaled raw data for comparing the accuracy.
 

Both Raw data and the Pca applied data gave almost the same accuracy which is almost 76 % whereas the scaled data’s accuracy is less when compared the above result which is around 70 %.


Question 2

Use pd_speech_features.csv

•	Importing all the necessary libraries and and taking the file as input.
•	Scaling the data after taking the entire dataset except class column as X and taking the class column as y and then Fitting the x data into the scaler method.
•	And then Applying PCA with 3 components and storing the data into a dataframe.
•	Applying SVM model over the scaled PCA applied dataset to find the accuracy of the model by taking the necessary X and Y variables.
•	Applying the SVM model to the raw data to find the data accuracy.
 
It seems that accuracy without pca is more than the accuracy when PCA is appled.


Question 3:

Apply Linear Discriminant Analysis (LDA) on Iris.csv dataset to reduce dimensionality of data to k =2.

•	 importing necessary libraires for executing the code.
•	importing the iris csv file into df variable and printing the top 5rows of the data frame.
•	Taking the dataframe, applying scaling to it and then fitting the data into LDA model to get the dimension reduced data.
•	Plotting the data which is discriminated.
 

Question 4 

Briefly identify the difference between PCA and LDA
•	PCA is an unsupervised learning method whereas LDA is a supervised learning method.
•	PCA is used for feature extraction and data compression, LDA is used for feature selection a