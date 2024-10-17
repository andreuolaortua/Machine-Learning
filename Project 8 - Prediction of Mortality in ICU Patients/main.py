import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Importing dataset
data = pd.read_csv('XXX.csv')
x = dataset.iloc[].values
y = dataset.iloc[].values

#Splitting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
sc = StandardScaler()
X_train = 
X_test = 