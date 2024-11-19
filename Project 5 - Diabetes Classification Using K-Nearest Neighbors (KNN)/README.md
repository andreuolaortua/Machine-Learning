# Diabetes Classification Using K-Nearest Neighbors (KNN)

This repository provides a complete pipeline for building a **K-Nearest Neighbors (KNN)** classifier to predict diabetes status based on a dataset of patient health metrics. The code covers data preprocessing, feature scaling, model training, hyperparameter tuning, and evaluation. 

---

## **Table of Contents**
1. [Dataset](#dataset)
2. [Dependencies](#dependencies)
3. [Code Overview](#code-overview)
4. [Key Results](#key-results)
5. [How to Run](#how-to-run)
6. [Conclusion](#conclusion)

---

## **Dataset**
The dataset is loaded from an external Excel file hosted online. It includes various health metrics for patients, such as:
- **Cholesterol**
- **Glucose**
- **BMI**
- **Systolic BP**
- **Diastolic BP**
- **Diabetes status** (target variable)

After preprocessing, the dataset is balanced for equal representation of patients with and without diabetes.

---

## **Dependencies**
This project requires the following Python libraries:
- `tqdm`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `requests`
- `io`

Install these libraries using:
```bash
pip install tqdm numpy pandas matplotlib seaborn scikit-learn requests
```

---

## **Code Overview**
### **1. Data Loading and Preprocessing**
- The dataset is fetched from a URL using the `requests` library.
- Unnecessary columns are dropped.
- Feature scaling is applied using `StandardScaler` to normalize numerical data.
- The dataset is balanced to ensure equal numbers of positive and negative diabetes cases.

### **2. Model Training**
- **K-Nearest Neighbors (KNN)** is used for classification.
- The dataset is split into training and testing sets using `train_test_split`.
- The target variable is encoded using `LabelEncoder`.

### **3. Hyperparameter Tuning**
- A grid search with cross-validation is performed to optimize the `n_neighbors` hyperparameter.

### **4. Evaluation**
- Metrics calculated include:
  - **Accuracy**
  - **Confusion Matrix**
- ANOVA feature selection is performed to rank feature importance.

### **5. Simpler Model**
- A simplified model is created using only the **Glucose** feature for comparison.

---

## **Key Results**
1. **Initial KNN Model Performance**
   - Best hyperparameters: `n_neighbors=7`
   - Accuracy: **88.46%**
2. **Simpler Model (Glucose Only)**
   - Accuracy: **91.67%**
3. **ANOVA Feature Selection**
   - Most important features (based on F-scores):
     1. **Glucose**
     2. **Chol/HDL ratio**
     3. **Cholesterol**
4. **Confusion Matrix**:
   ```
   [[ 8  8]
    [ 1 61]]
   ```

---

## **How to Run**
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Run the script:
   ```bash
   python diabetes_classification.py
   ```

---

## **Conclusion**
This project demonstrates the implementation of a KNN classifier to predict diabetes, highlighting the impact of preprocessing, hyperparameter tuning, and feature selection. The simplified model using only glucose achieves comparable accuracy, showing the potential for single-feature prediction in this case.