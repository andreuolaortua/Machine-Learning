# Drug Classification Using Decision Tree

## Project Overview

This project uses a **Decision Tree Classifier** to predict the type of drug prescribed to a patient based on various medical attributes such as **age**, **sex**, **blood pressure**, **cholesterol levels**, and **sodium-to-potassium ratio**. The goal is to classify patients into different drug categories.

## Requirements

To run this project, the following libraries are required:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **scikit-learn**: For machine learning algorithms and evaluation metrics.

You can install the necessary libraries using pip:

```bash
pip install pandas numpy scikit-learn
```

## Dataset

The dataset contains the following features:

- **Age**: Age of the patient (numeric).
- **Sex**: Gender of the patient (0 for male, 1 for female).
- **BP (Blood Pressure)**: Blood pressure level (encoded as low, normal, or high).
- **Cholesterol**: Cholesterol level (encoded as normal or above normal).
- **Na_to_K**: Sodium-to-potassium ratio (numeric).

### Target Variable:
- **Drug**: The drug prescribed to the patient. The target variable includes categories like **drugA**, **drugB**, **drugC**, **drugX** and **drugY**.

## Model

In this project, a **Decision Tree Classifier** is used to predict the drug prescribed based on the input features. The classifier is trained on the data and evaluated using accuracy metrics.

### Key Components:

- **Data Preprocessing**: The dataset is preprocessed by encoding categorical variables such as **Sex**, **BP**, and **Cholesterol** using **LabelEncoder** from `sklearn`.
  
- **Model Training**: The **Decision Tree Classifier** is trained using the features to predict the drug type. The model is evaluated using the **accuracy score**.

## Evaluation

The model's performance is evaluated using the **accuracy score**. The model achieved a perfect score on the test data, showing high accuracy in predicting the drug categories.

```
Accuracy:  1.0
```

## Conclusion

The **Decision Tree Classifier** performs excellently for this classification task, providing an accuracy of **100%**. Further improvements can be made by exploring different machine learning algorithms or fine-tuning the model parameters.
