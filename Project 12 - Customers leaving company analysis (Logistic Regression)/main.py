#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score, log_loss
import seaborn as sns

#Reading dataset
dataset = pd.read_csv('ChurnData.csv')
print(dataset.head())

#Preprocessing and selecting data
dataset['churn'] = dataset['churn'].astype('int')
X = np.asanyarray(dataset[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asanyarray(dataset[['churn']])

#Scaling data
sc = StandardScaler()
X = sc.fit_transform(X)
print(X[0:5])


#Splitting the dataset into test and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=4)

#Modeling Logistic Regression
log_reg = LogisticRegression(C=0.01, solver='liblinear')
log_reg.fit(X_train, y_train)

#Predicting the results
yhat = log_reg.predict(X_test)
y_hat_proba = log_reg.predict_proba(X_test) # returns estimates for all classes, ordered by the label of classes. First column is the probability of class 0, P(Y=0|X), and second column is probability of class 1, P(Y=1|X)

#Evaluating model
jaccard_score(y_test, yhat, pos_label=0)


# Building and visualizing the confusion matrix
"""
cm = confusion_matrix(y_test, yhat)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["churn=0", "churn=1"], yticklabels=["churn=0", "churn=1"])
plt.title("Matriz de Confusión")
plt.xlabel("Prediction label")
plt.ylabel("True label")
plt.show()
"""

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

#Classification report
print(classification_report(y_test, yhat))

#Log loss
print(log_loss(y_test, y_hat_proba))


plt.show()