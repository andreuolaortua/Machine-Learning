# Breast Cancer Diagnosis with SVM

This project focuses on classifying breast cancer cells as benign or malignant using a Support Vector Machine (SVM) model. The dataset, `cell_samples.csv`, contains features extracted from cell samples, which are used to train and evaluate the classification model.

---

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Features](#features)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## Installation

Ensure you have Python 3.7+ installed. Install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Dataset

The `cell_samples.csv` dataset contains information about cell samples, including measurements for different features such as clump thickness, uniformity of size and shape, and others.

### Data Overview

| Column      | Type    | Description                          |
|-------------|---------|--------------------------------------|
| ID          | int     | Unique identifier for the sample     |
| Clump       | int     | Clump thickness                      |
| UnifSize    | int     | Uniformity of cell size              |
| UnifShape   | int     | Uniformity of cell shape             |
| MargAdh     | int     | Marginal adhesion                    |
| SingEpiSize | int     | Single epithelial cell size          |
| BareNuc     | object  | Bare nuclei (converted to numeric)   |
| BlandChrom  | int     | Bland chromatin                      |
| NormNucl    | int     | Normal nucleoli                      |
| Mit         | int     | Mitoses                              |
| Class       | int     | Diagnosis (2 = benign, 4 = malignant)|

---

## Features

1. **Data Cleaning**: Non-numeric values in the `BareNuc` column are removed, and the column is converted to integers.
2. **Data Splitting**: The dataset is split into training (75%) and testing (25%) sets.
3. **Model Training**: A Support Vector Machine (SVM) classifier with an RBF kernel is trained on the training set.
4. **Evaluation Metrics**:
    - Confusion Matrix
    - Classification Report
    - Jaccard Index
    - F1-Score

---

## Usage

1. Clone the repository and navigate to the project directory.
2. Place the `cell_samples.csv` file in the same directory.
3. Run the script:

   ```bash
   python breast_cancer_svm.py
   ```

4. The script will:
    - Load and preprocess the dataset.
    - Train an SVM model.
    - Evaluate the model with metrics like accuracy, precision, recall, and F1-score.
    - Plot the confusion matrix.

---

## Results

### Model Evaluation
The SVM model achieved the following metrics on the test set:

- **Accuracy**: 95%
- **F1-Score**: 0.9479
- **Jaccard Index**: 0.9189
- **ANOVA Feature Selection** is performed to rank feature importance.
   - Most important features (based on F-scores):
     1. **BareNuc**
     2. **UnifShape**
     3. **UnifSize**
     
### Confusion Matrix

|               | Predicted Benign (2) | Predicted Malignant (4) |
|---------------|-----------------------|--------------------------|
| **Actual Benign (2)**   | 102                   | 8                        |
| **Actual Malignant (4)**| 1                     | 60                       |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Benign (2)  | 0.99      | 0.93   | 0.96     | 110     |
| Malignant (4)| 0.88      | 0.98   | 0.93     | 61      |

---

## License

This project is licensed under the MIT License.