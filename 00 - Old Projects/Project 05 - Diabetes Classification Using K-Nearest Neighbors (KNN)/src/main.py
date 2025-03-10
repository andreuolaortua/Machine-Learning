from tqdm import tqdm
import numpy as np
import requests
import pandas as pd
from io import BytesIO
####################from itertools import accumulate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.feature_selection import f_classif
from sklearn.utils import resample

# Suppress warnings generated by the code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

sns.set_context('notebook')
sns.set_style('white')

#Load the data
response = requests.get('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/X4i8vXLw81g4wEH473zIFA/Diabetes-Classification.xlsx', verify=False)
response.raise_for_status()

df = pd.read_excel(BytesIO(response.content))
print(df.head())
df = df.drop(columns=['Unnamed: 16', 'Unnamed: 17']) #Elimination of the last two unusable columns
print(df.head())


#checking proportion of patients with and without diabetes
frequency_table = df['Diabetes'].value_counts()
props = frequency_table.apply(lambda x: x / len(df['Diabetes']))
print(props)

#Feature scaling
df_reduced = df[["Diabetes", "Cholesterol", "Glucose", "BMI", "Waist/hip ratio", "HDL Chol", "Chol/HDL ratio", "Systolic BP", "Diastolic BP", "Weight"]]

numerical_columns = df_reduced.iloc[:, 1:10]

# Applying scaling
scaler = StandardScaler()
preproc_reduced = scaler.fit(numerical_columns)
df_standardized = preproc_reduced.transform(numerical_columns)

# Converting the standardized array back to DataFrame
df_standardized = pd.DataFrame(df_standardized, columns=numerical_columns.columns)

print(df_standardized.describe())

df_stdize = pd.concat([df_reduced['Diabetes'], df_standardized], axis=1)
print(df_stdize)

#Split the data set
X = df_stdize.drop(columns=['Diabetes'])
y = df_stdize['Diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)


# Create a KNN classifier
knn = KNeighborsClassifier()

knn.fit(X_train, y_train_encoded)

#calculate overall accuracy
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Accuracy: {accuracy:.2%}')
param_grid = {'n_neighbors': range(1, 12)}

# Perform grid search with cross-validation
grid_search = GridSearchCV(knn, param_grid, cv=10)
grid_search.fit(X_train, y_train_encoded)


# Best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print(f"Best accuracy score: , {grid_search.best_score_:.3f}")

# Full results
results = grid_search.cv_results_
for mean_score, std_score, params in zip(results['mean_test_score'], results['std_test_score'], results['params']):
    print(f"Mean accuracy: {mean_score:.3f} (std: {std_score:.3f}) with: {params}")

#ANOVA for feature selection
fs_score, fs_p_value = f_classif(X, y)

# Combine scores with feature names
fs_scores = pd.DataFrame({'Feature': X.columns, 'F-Score': fs_score, 'P-Value': fs_p_value})
fs_scores = fs_scores.sort_values(by='F-Score', ascending=False)

print(fs_scores)


# Converting Diabetes column into binary (0 for No Diabetes and 1 for Diabetes)
df_stdize['Diabetes'] = np.where(df_stdize['Diabetes'] == 'Diabetes', 1, 0)
df_stdize


# Number of rows for positive diabetes
positive_diabetes = df_stdize[df_stdize['Diabetes'] == 1].shape[0]
print('Number of rows for positive diabetes: ', positive_diabetes)

# Sample negative cases to match positive cases
negative_diabetes = df_stdize[df_stdize['Diabetes'] == 0]
negative_diabetes_downsampled = resample(negative_diabetes, replace=False, n_samples=positive_diabetes, random_state=42)

# Put positive and negative diabetes case into one df -> balanced
balanced = pd.concat([negative_diabetes_downsampled, df_stdize[df_stdize['Diabetes'] == 1]])
print(balanced.sample(5))

print(balanced['Diabetes'].value_counts())

#Fitting on simpler model
X_simple = balanced[['Glucose']]
y = balanced['Diabetes']

# Split the data
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y, test_size=0.2, random_state=42)

knn_simple = KNeighborsClassifier()
knn_simple.fit(X_train_simple, y_train_simple)
y_pred_simple = knn_simple.predict(X_test_simple)
accuracy = accuracy_score(y_test_simple, y_pred_simple)
print(f'Accuracy: {accuracy:.2%}')


# Evaluate confusion matrix
cm = confusion_matrix(y_test_encoded, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(cm)
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Accuracy: {accuracy:.2%}')