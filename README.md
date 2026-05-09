# electrical-fault-detection-ml
Electrical fault detection in 3-phase systems using machine learning. Improved classification accuracy from 88.7% to 95.9% through feature engineering (impedance, power, magnitude) and comparison of ensemble models.
# Electrical Fault Detection in 3-Phase Systems using Machine Learning

## Overview
This project focuses on detecting electrical faults in a 3-phase system using machine learning. 

A baseline model was built using raw voltage and current signals, then improved using feature engineering based on electrical principles.

## Dataset
The dataset used in this project was sourced from Kaggle:

https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification/data

This dataset contains line voltages and currents for different fault conditions in a 3-phase electrical system. 

Note: The dataset is not included in this repository. Please download it from the link above and place it in the project directory before running the code.

## Key Idea
Instead of relying only on raw data, additional features were created:
- Impedance (approximation)
- Real Power (P)
- Voltage magnitude (V_mag)
- Current magnitude (I_mag)
- Apparent Power (S)

These features improved model performance significantly.

## Results

| Model                        | Accuracy |
|-----------------------------|----------|
| Baseline (Random Forest)    | 88.7%    |
| Random Forest (Engineered)  | 95.3%    |
| Gradient Boosting           | 94.9%    |
| XGBoost                     | **95.9%** |

## Feature Importance
The most important predictors were:
- Impedance (Z_approx)
- Current magnitude (I_mag)
- Voltage magnitude (V_mag)

This aligns with electrical system behavior, where faults strongly affect impedance and current.

## Tools Used
- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn

## Conclusion
Feature engineering significantly improved fault detection accuracy.  
Among tested models, XGBoost performed best.  
