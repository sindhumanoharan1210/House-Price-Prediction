# housing-price-Prediction

Predicting California median house values using machine learning regression techniques on the 
California Housing dataset. The project covers end-to-end ML workflow — EDA, preprocessing, 
model comparison via cross-validation, hyperparameter tuning, and a deployable predictive system.

---

## Dataset

- **Source:** California Housing Dataset (`housing.csv`)
- **Records:** 20,640 | **Features:** 9 (8 numerical, 1 categorical)
- **Target:** `median_house_value`
- **Missing Values:** `total_bedrooms` — 207 nulls (handled via median imputation)

---

## Project Workflow

1. **Exploratory Data Analysis** — distribution plots, outlier detection (boxplots), correlation heatmap
2. **Preprocessing Pipeline** — median imputation + standard scaling for numerical features; 
   mode imputation + one-hot encoding for categorical features (`ocean_proximity`)
3. **Model Comparison** — 5-Fold Cross-Validation across 5 models
4. **Hyperparameter Tuning** — GridSearchCV on the best model (243 candidate combinations)
5. **Evaluation** — RMSE, MAE, R² on train and test sets; residual analysis
6. **Predictive System** — single-record prediction function using the trained pipeline

---

## Models Compared (5-Fold CV)

| Model | CV RMSE | CV MAE | CV R² |
|---|---|---|---|
| **HistGradientBoosting** | **48,267** | **32,368** | **0.826** |
| Random Forest | 49,363 | 32,220 | 0.818 |
| Ridge | 68,596 | 49,664 | 0.648 |
| Lasso | 68,603 | 49,667 | 0.648 |
| Linear Regression | 68,604 | 49,667 | 0.648 |

---

## Final Model Performance (Tuned HistGradientBoostingRegressor)

| Split | RMSE | MAE | R² |
|---|---|---|---|
| Train | 41,284 | 28,198 | 0.873 |
| **Test** | **47,373** | **31,614** | **0.829** |

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Techniques:** Pipeline, ColumnTransformer, KFold CV, GridSearchCV

---

## Project Structure
housing-price-regression-hgb/
│
├── housing.csv                        # Dataset
├── housing_price_prediction.ipynb     # Main notebook
└── README.md
---

## Key Findings

- `median_income` is the strongest predictor of house value (correlation: 0.688)
- `total_rooms`, `total_bedrooms`, `households`, and `population` are highly multicollinear
- The target variable is right-skewed with a hard cap at $500,001 (965 records), indicating 
  censored data — this contributes to residual spread at higher predicted values
- Tuned HGB outperforms linear models by ~30% in RMSE

---

## Usage

```python
example_pred = predict_house_price(
    model=pipeline,
    longitude=-122.23,
    latitude=37.88,
    housing_median_age=41,
    total_rooms=880,
    total_bedrooms=129,
    population=322,
    households=126,
    median_income=8.3252,
    ocean_proximity="NEAR BAY"
)
print("Predicted House Value:", round(example_pred, 2))
# Output: 448545.09
```
