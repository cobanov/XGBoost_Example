# XGBoost_Example

This repository contains an example for XGBoost


##### Author: Mert Cobanoglu





```python

l1_params = [1, 10, 100]
rmes_l1 = []

for reg in l1_params:
    params["alpha"] = re
    cv_reults = xgb.cv(dtrain= , params=params, nfold=4, num_boost_round=10, metrics="rmse", as_pandas=True, seed=123)
    rmses_l1.append(cv_results["test-rmse-mean"].tail(1).values[0])

```
