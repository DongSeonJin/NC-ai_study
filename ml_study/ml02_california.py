from sklearn.svm import LinearSVR, SVR
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.7,
    random_state=72,
    shuffle=True
)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# 2. 모델
# model = LinearSVR()
model = SVR()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
result = model.score(x_test, y_test)
print('R2 score : ', result)
# 회귀모델의 model.score값은 r2 score
# LinearSVR R2 score : -0.3813703689476129
# SVR r2 score : -0.041213036816662996
