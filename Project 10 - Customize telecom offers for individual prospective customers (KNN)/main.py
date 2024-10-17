#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#Importing libraries
df = pd.read_csv('teleCust1000t.csv')
df.head()

#See how many of each class is in our data set
df['custcat'].value_counts()

#Exploring data
df.hist(column='income', bins=50)

#Define feature sets X
df.columns

#X = df.iloc[:, 0:11].values
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]
y = df['custcat'].values
y[0:5]

#Normalizing data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

#Splitting into train set and test set
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Training
k=4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

#Predicting
yhat = neigh.predict(X_test)
yhat[0:5]

#Accuracy evaluating
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
