# Electricity Consumption Forecasting & Anomaly Detection

## Problem Statement
Four models were developed and evaluated for household electricity consumption prediction. Random Forest was selected as the preferred model based on its lowest RMSE of 0.2192. However, the Baseline model performed best in terms of MAE and MAPE.
The selected Random Forest model and the model comparison results were saved for future use. Further work could include hyperparameter tuning, additional time-based features, and verification that all lag and rolling features are generated without future-data leakage.
The reported metrics represent the results of the current experiment and may change if preprocessing, feature engineering, or model configurations are modified.

## Dataset
The household power consumption dataset is loaded into the Google Colab environment and inspected to understand its structure, column names, data types, and missing values.
The dataset includes time information and household electricity measurements. The target variable for prediction is provisionally identified as Global_active_power, subject to confirmation by the project team.
Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54

## Methodology
The project used the Household Electric Power Consumption dataset to predict household electricity consumption. The target variable was Global_active_power.
After preprocessing and feature engineering, 2,015,236 rows were available for modeling, with zero duplicate timestamps reported in the preceding data validation. The features included hour of day, day of week, month, previous consumption values, and a rolling mean of recent consumption.

## Results
Model Performance
The following results were obtained from the Colab experiments.
Model
MAE

RMSE

MAPE (%)

Baseline

0.0722

0.2223

7.2349

Linear Regression

0.0846

0.2201

10.0700

Random Forest

0.0820

0.2192

8.7925

Keras Neural Network

0.0959

0.2231

12.4322.

## App Demo
(link and screenshot to be added)

## How to Run
(to be written)

## Limitations and Future Work
The Random Forest Regressor achieved the lowest RMSE of 0.2192 among the four models. It was therefore selected as the best model according to the project's primary selection metric.
The Baseline model achieved the lowest MAE of 0.0722 and the lowest MAPE of 7.2349%. This shows that the previous-observation prediction was a strong benchmark and performed better than the more complex models on average absolute and percentage error.
Linear Regression achieved an RMSE of 0.2201, slightly higher than Random Forest. The Keras Neural Network produced the highest MAE, RMSE, and MAPE in this experiment, indicating that it did not outperform the other approaches under the configurations tested.
The results demonstrate that model complexity alone does not guarantee improved predictive performance. The baseline should remain an important reference when interpreting the benefits of more advanced models.

## Team
(names and roles)
