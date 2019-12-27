# XGBoost_Example

This repository contains an example for XGBoost


##### Author: Mert Cobanoglu

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

### XGBoost

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

### L1 Reg
```python
l1_params = [1, 10, 100]
rmes_l1 = []

for reg in l1_params:
    params["alpha"] = re
    cv_reults = xgb.cv(dtrain= , params=params, nfold=4, num_boost_round=10, metrics="rmse", as_pandas=True, seed=123)
    rmses_l1.append(cv_results["test-rmse-mean"].tail(1).values[0])
```
