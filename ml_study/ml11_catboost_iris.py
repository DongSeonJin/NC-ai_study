import numpy as np
from catboost import CatBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# Scaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

n_splits = 7
random_state = 72
# kflod
kflod = StratifiedKFold(
    n_splits=n_splits,
    random_state=random_state,
    shuffle=True
)

# 2. 모델
model = CatBoostClassifier()

# 3. 훈련, 평가

score = cross_val_score(
    model,
    x, y,
    cv=kflod
)
print('acc score : ', score, '\n cros_val_score', round(np.mean(score), 4))

# acc score :  [1.         0.90909091 0.95454545 0.95238095 0.85714286 1.0.95238095]
#  cros_val_score 0.9465