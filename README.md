<p align="center">
  <img src="https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/xgboost.png" width=300>
</p>

<p align="center">This repository contains an example for XGBoost
<b> - Author: Mert Cobanoglu</b> </p>

```python
import xgboost as xgboost
import pandas as pd

churn_data = pd.read_csv("classification_data.csv")

churn_dmatrix = xgb.DMatrix(data=churn_data.iloc[:, -1],
                            label=churn_data.month_5_still_here)

params = {"objective":"binary:logistic", max_depth=4}

cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=4,
                    num_boost_round=10, metrics="error", as_pandas=True)

```

## Data Prep

### Label Encoding

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
import pandas as pd

cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"] 
data = pd.read_csv("iris.data", names=cols)

label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(data["class"])

or

for cols in data.columns:
    data[cols] = label_encoder.fit_transform(data[cols])
```

### One Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder
oh_encoder = OneHotEncoder(sparse=False)
targets = targets.reshape(150, 1)
oneho = oh_encoder.fit_transform(targets)
```

## Optimization

### Cross-Validation with Early Stopping

```python
# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix,
                    params=params,
                    early_stopping_rounds=10,
                    num_boost_round=50,
                    seed=123,
                    metrics="rmse",
                    nfold=3,
                    as_pandas=True)

# Print cv_results
print(cv_results)
```

### L1 Reg

```python
l1_params = [1, 10, 100]
rmes_l1 = []

for reg in l1_params:
    params["alpha"] = reg
    cv_reults = xgb.cv(dtrain=data, params=params, nfold=4,
                       num_boost_round=10, metrics="rmse",
                       as_pandas=True, seed=123)
    rmses_l1.append(cv_results["test-rmse-mean"].tail(1).values[0])
```

## Fine-Tuning

### Grid Search

```python
housing_dmatrix = xgb.DMatrix(data=X, label=y)

gbm_param_grid = {'colsample_bytree': [0.3, 0.7],
                  'n_estimators': [50],
                  'max_depth': [2, 5]}

gbm = xgb.XGBRegressor()
grid_mse = GridSearchCV(param_grid=gbm_param_grid,
                        estimator=gbm,
                        scoring="neg_mean_squared_error",
                        cv=4, verbose=1)
grid_mse.fit(X, y)

print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))
```

### Random Search

```python
gbm_param_grid = {'n_estimators': [25],
                  'max_depth': range(2, 12)}

gbm = xgb.XGBRegressor(n_estimators=10)

randomized_mse = RandomizedSearchCV(param_distributions=gbm_param_grid,
                                    estimator=gbm, scoring="neg_mean_squared_error",
                                    n_iter=5, cv=4, verbose=1)
randomized_mse.fit(X, y)

print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))
```
### Pipeline

```python
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

X.LotFrontage = X.LotFrontage.fillna(0)

steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor())]

xgb_pipeline = Pipeline(steps)
xgb_pipeline.fit(X.to_dict("records"), y)
```

### Visualization

```python
from xgboost import plot_importance, plot_tree
import graphviz
import matplotlib.pyplot as plt

xgb.plot_importance(xg_reg)
xgb.plot_tree(xg_reg)

```

## Parameters

**learning_rate:** step size shrinkage used to prevent overfitting. Range is [0,1]

**max_depth:** determines how deeply each tree is allowed to grow during any boosting round.

**subsample:** percentage of samples used per tree. Low value can lead to underfitting.

**colsample_bytree:** percentage of features used per tree. High value can lead to overfitting.

**n_estimators:** number of trees you want to build.

**objective:** determines the loss function to be used like reg:linear for regression problems, reg:logistic for classification problems with only decision, binary:logistic for classification problems with probability.

### Reg. Parameters

**gamma:** controls whether a given node will split based on the expected reduction in loss after the split. A higher value leads to fewer splits. Supported only for tree-based learners.

**alpha:** L1 regularization on leaf weights. A large value leads to more regularization.

**lambda:** L2 regularization on leaf weights and is smoother than L1 regularization.
