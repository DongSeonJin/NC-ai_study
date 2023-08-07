from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
# 1. 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, 
#     train_size=0.7,
#     test_size=0.3,
#     random_state=72,
#     shuffle=True
# )

# Scaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

# KFlod
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=72
)

# 2. 모델
model = SVR()

# 3. 훈련, 평가
score = cross_val_score(
    model,
    x, y, 
    cv = kfold
)

print('r2 score : ', score, '\n cross_val_score : ', round(np.mean(score), 4))
# scaler 미적용
# r2 score :  [-0.01320356 -0.01042838 -0.03806029 -0.03233878 -0.02485869] 
#  cross_val_score :  -0.0238

# scaler 적용 
# r2 score :  [0.74977437 0.72822127 0.73631372 0.75289926 0.72902466] 
#  cross_val_score :  0.7392