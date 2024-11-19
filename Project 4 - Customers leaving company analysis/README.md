# Customer Churn Prediction - Logistic Regression

## Overview

This project uses **Logistic Regression** to predict customer churn (the likelihood that a customer will leave the company). The model is trained on various features, such as customer tenure, age, income, and others, to predict whether the customer will churn or not. We evaluate the model's performance using multiple metrics such as **confusion matrix**, **Jaccard score**, and **log loss**.

## Requirements

- Python 3.x
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

1. Clone this repository:
    ```bash
    git clone <REPOSITORY_URL>
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1. Make sure the dataset `ChurnData.csv` is in the correct directory.
2. Run the main script:
    ```bash
    python churn_prediction.py
    ```

## Workflow

1. **Loading the dataset**: The dataset `ChurnData.csv` is read, containing information about customer details and whether they have churned.

2. **Preprocessing**:
   - The `churn` column is converted to an integer.
   - Features are selected, and the data is scaled using `StandardScaler`.

3. **Modeling**:
   - A **Logistic Regression** model is trained on the data.
   - Predictions are made on the test set.

4. **Evaluation**:
   - The model is evaluated using several metrics:
     - **Jaccard score** for classification performance.
     - **Confusion matrix** to visualize true positives, false positives, true negatives, and false negatives.
     - **Classification report** to evaluate precision, recall, and F1-score.
     - **Log loss** to assess the probability accuracy of the model.

5. **Visualization**:
   - A non-normalized **confusion matrix** is plotted to visualize the true vs predicted labels.

## Example Results

- **Confusion Matrix**:
    ```
    [[ 6  9]
     [ 1 24]]
    ```

- **Classification Report**:
    ```
              precision    recall  f1-score   support

           0       0.73      0.96      0.83        25
           1       0.86      0.40      0.55        15

    accuracy                           0.75        40
    macro avg      0.79      0.68      0.69        40
    weighted avg   0.78      0.75      0.72        40
    ```

- **Log Loss**: 0.60

## Conclusion

The Logistic Regression model performs reasonably well in predicting customer churn. The precision and recall vary across classes, with a strong recall for churn=0 (non-churn) but lower for churn=1 (churn). The log loss indicates moderate performance in terms of predicted probabilities.
