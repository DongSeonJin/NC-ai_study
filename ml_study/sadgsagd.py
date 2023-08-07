import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=72,
    shuffle=True
)

# Scaler
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# KFlod
n_splits = 7
random_state = 72
kflod = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state
)

# 2. 모델구성
model = DecisionTreeClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
score = cross_val_score(
    model,
    x, y,
    cv = kflod
)

print('acc score : ', score, '\n cross_val_score : ', round(np.mean(score), 4))

# n_splits = 7, SVC
# acc score :  [1.         0.86363636 1.         0.95238095 0.9047619  0.952380951.        ]
# cross_val_score :  0.9473

# n_splits = 7, LinearSVC
# acc score :  [0.95454545 0.90909091 1.         0.9047619  0.9047619  0.809523811.        ]
# cross_val_score :  0.9397

n_features = datasets.data.shape[1]
plt.barh(range(n_features), model.feature_importances_,align='center')
plt.yticks(np.arange(n_features), datasets.feature_names)
plt.title('Cancer Feature Importances')
plt.ylabel('Feature')
plt.xlabel('Importances')
plt.ylim(-1, n_features)
plt.show()