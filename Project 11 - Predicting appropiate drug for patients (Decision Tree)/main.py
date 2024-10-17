#importing libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Reading dataset
dataset = pd.read_csv('drug200.csv')
#print(dataset.head())
#print(dataset.shape)

#Encoding dataset
le = LabelEncoder()
dataset['Sex'] = le.fit_transform(dataset['Sex'])
dataset['BP'] = le.fit_transform(dataset['BP'])
dataset['Cholesterol'] = le.fit_transform(dataset['Cholesterol'])
#dataset['Drug'] = le.fit_transform(dataset['Drug'])


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#Another way to classify
# X = dataset[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
# y = dataset[['Drug']].values
print(X[:5, :])

#splitting dataset into trainset and testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) #Ensuring reproducibility using the random_state parameter

#Another way to split the data
# msk = np.random.rand(dataset) < 0.8
# X = dataset[msk]
# y = dataset[~msk]
"""
#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit(X_test)
"""
#print(X_train)

#Modeling
dtree = DecisionTreeClassifier(criterion='entropy', max_depth = 4)
dtree.fit(X_train, y_train)

#Predicting
y_pred = dtree.predict(X_test)
print(y_pred)
print(y_test)

#Evaluation
print("Accuracy: ", accuracy_score(y_test, y_pred))


