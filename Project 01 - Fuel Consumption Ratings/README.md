
# CO₂ Emissions Prediction with Regression Models

## Project Overview

This project analyzes vehicle data to predict **CO₂ emissions** using various regression techniques, including:

- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression

The dataset contains attributes such as engine size, cylinders, and fuel consumption, which are used as features to estimate emissions.

## Requirements

To run this project, you need the following Python libraries:

- **pandas**: For data manipulation and exploration.
- **numpy**: For numerical operations.
- **matplotlib**: For data visualization.
- **scikit-learn**: For regression models and evaluation metrics.

Install the required libraries with:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Dataset

The dataset contains information on vehicle specifications and their corresponding **CO₂ emissions**.
### Features Used in Modeling

- **Engine Size**: Size of the engine in liters.
- **Cylinders**: Number of engine cylinders.
- **Fuel Consumption (City, Highway, Combined)**: Fuel efficiency in different driving conditions.
- **Transmission**: Transmission type.
- **Fuel Type**: Fuel classification.
- **CO₂ Emissions**: Target variable, measured in grams per kilometer.

The dataset contains additional features like `MAKE` and `MODEL` that provide descriptive metadata about the vehicles.

## Models and Methods

### 1. Data Exploration

- Visualized relationships between features and **CO₂ emissions**:
  - Engine size vs CO₂ emissions
  - Cylinders vs CO₂ emissions
  - Combined fuel consumption vs CO₂ emissions
- Created histograms for key features to understand data distribution.

### 2. Simple Linear Regression

- Predicted **CO₂ emissions** using **engine size** as the sole predictor.
- **Model Coefficients**:
  - Coefficients: `39.04`
  - Intercept: `125.78`
- **Evaluation**:
  - Mean Absolute Error (MAE): `22.09`
  - Mean Squared Error (MSE): `835.98`
  - R² Score: `0.79`

### 3. Multiple Linear Regression

- Used **engine size**, **cylinders**, and **combined fuel consumption** as predictors.
- **Model Coefficients**:
  - `9.91` (Engine Size), `7.61` (Cylinders), `9.92` (Fuel Consumption)
- **Evaluation**:
  - Mean Squared Error (MSE): `575.13`
  - Variance Score (R²): `0.85`

### 4. Polynomial Regression

- Modeled a third-degree polynomial relationship between **engine size** and **CO₂ emissions**.
- **Model Coefficients**:
  - Coefficients: `[34.29, 3.05, -0.39]`
  - Intercept: `124.34`
- **Evaluation**:
  - Mean Absolute Error (MAE): `22.17`
  - Mean Squared Error (MSE): `831.54`
  - R² Score: `0.79`

### Visualization

- Scatter plots with regression lines for simple and polynomial regression.
- Histogram of dataset features for initial exploration.

## Results and Observations

- **Multiple Linear Regression** outperformed other models with an R² score of **0.85**, indicating it captures the relationships in the data better than simple or polynomial regression.
- **Polynomial Regression** slightly improved on simple regression but did not outperform multiple regression in this case.
- The dataset's attributes, especially engine size and fuel consumption, significantly influence CO₂ emissions.

## Conclusion

This project demonstrated the application of regression techniques for predicting CO₂ emissions. 
The results suggest that using multiple features improves the model's predictive power. 

