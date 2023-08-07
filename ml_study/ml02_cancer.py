from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# [실습] SVC와 LinearSVC 모델을 적용하여 코드를 완성하시오.

datasets = load_breast_cancer()
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
model = LinearSVC()
# model = SVC()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
result = model.score(x_test, y_test)
print('result : ', result)  # 분류모델은 score = accuracy

# LinearSVC = 0.5889724310776943
# SVC = 0.9147869674185464




