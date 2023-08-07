from sklearn.utils import all_estimators
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MaxAbsScaler

# 1. 데이터 

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7,
    random_state=72, shuffle=True
)

print(x_train.shape, y_train.shape) # (105, 4) (105,)
print(x_test.shape, y_test.shape) # (45, 4) (45,)

# Scaler 적용
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
allAlgorithms = all_estimators(type_filter='classifier')
print('갯수', len(allAlgorithms)) 

# 3. 훈련
for(name, algorithm) in allAlgorithms : 
    try : 
        model = algorithm()
        model.fit(x_train, y_train)
        result = model.score(x_test, y_test)
        print(name, '의 정답률 : ', result)
    except : 
        print(name, '안나온 것')