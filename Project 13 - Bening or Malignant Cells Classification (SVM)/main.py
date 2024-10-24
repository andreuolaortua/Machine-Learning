#Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_score
import matplotlib.pyplot as plt

#Reading data
dataset = pd.read_csv('cell_samples.csv')
print(dataset.head())
print(dataset.dtypes)

dataset = dataset[pd.to_numeric(dataset['BareNuc'], errors='coerce').notnull()]
dataset['BareNuc'] = dataset['BareNuc'].astype('int')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


#Splitting the test and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 4)


#Modeling (SVM with scikit-learn)
svm_m = svm.SVC(kernel='rbf')
svm_m.fit(X_train, y_train)

#Predicting the results
yhat = svm_m.predict(X_test)


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



# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
print(cnf_matrix)
np.set_printoptions(precision=2)

#Classification report
print(classification_report(y_test, yhat))

#Evaluating model
print(jaccard_score(y_test, yhat, pos_label=2))
print(f1_score(y_test, yhat, average='weighted'))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')


plt.show()