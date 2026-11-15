# Appliances Energy Prediction using Machine Learning

Predicts household appliance energy consumption (in Wh) from indoor sensor readings, outdoor weather and time-of-day features, using classical ML regressors and neural networks.

## Dataset
- 19,735 records at 10-minute intervals, 29 original columns
- Target: `Appliances`
- Preprocessing: dropped `rv1`/`rv2`, derived `Month` and `Hour` from `date`, leaving 27 input features
- No missing values or duplicates; outliers inspected but not removed

## Approach
- **EDA:** hourly consumption trends, histograms, box plots, correlation heatmap
- **ML models:** Linear Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVR, evaluated on 5 train-test splits (60:40 to 80:20)
- **ANN:** 10 Keras configurations (Adam vs SGD, varying depth and width)
- **Tuning:** GridSearchCV (5-fold CV) for all seven models
- **Feature selection:** VIF-based removal of multicollinear features (27 → 19), compared against using all features
- **Metrics:** R², RMSE, MAE, MAPE

## Key Results (80:20 split, all 27 features)
| Model | R² | RMSE | MAE | MAPE (%) |
|---|---|---|---|---|
| **KNN (GridSearchCV)** | **0.6238** | **58.29** | **26.54** | **25.52** |
| Random Forest (GridSearchCV) | 0.5975 | 60.29 | 29.26 | 29.86 |
| XGBoost (best untuned, 75:25) | 0.5475 | 67.04 | 33.28 | 34.70 |
| Best ANN (256-128-64-32, Adam) | 0.5162 | 66.10 | 31.59 | 31.46 |

## Findings
- Tuned KNN was the best model overall, beating tree ensembles and every ANN.
- Hyperparameter tuning improved almost every model.
- VIF feature selection did not improve accuracy; it gave a simpler feature set at similar untuned performance.
- Individual features correlate weakly with consumption (strongest: `Hour`, 0.22; `lights`, 0.20), so appliance usage is only moderately predictable.

## Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, TensorFlow/Keras, statsmodels, Matplotlib
