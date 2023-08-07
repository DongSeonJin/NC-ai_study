import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# 1. 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7,
    random_state=72,
    shuffle=True
)

# Scaler 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# kfold 
kfold = KFold(
    n_splits=5,
    random_state=72,
    shuffle=True
)

# 2. 모델

cat_model = CatBoostRegressor()
lgbm_model = LGBMRegressor()
xgb_model = XGBRegressor()

model = VotingRegressor(
    estimators=[
        ('cat_model', cat_model),
        ('lgbm_model', lgbm_model),
        ('xgb_model', xgb_model)
    ],
    weights=[1, 1, 1],
    n_jobs=-1
)

# 3. 훈련

model.fit(x_train, y_train)

# 4. 평가

classifiers = [cat_model, lgbm_model, xgb_model]

for classifier in classifiers:
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    class_name = classifier.__class__.__name__
    print('{0} MSE: {1:.4f}'.format(class_name, mse))

result = model.score(x_test, y_test)
print('Voting 결과: ', result)