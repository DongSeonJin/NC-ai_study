# [실습]
# 1. r2 score룰 음수가 아닌 0.5이하로 만드세요.
# 2. 데이터는 건드리지 마세요.
# 3. 레이어는 인풋, 아웃풋 포함 7개 이상 만드세요.
    # (히든레이어가 5개 이상이어야 됨)
# 4. batch_size=1 이어야 함
# 5. 히든레이어의 노드(뉴런) 갯수는 10이상 100개 이하로 하세요.
# 6. train_size=0.7로 하세요.
# 7. epochs=100 이상으로 하세요.

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17 ,18 ,19, 20 ])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17 ,18 ,19, 20 ])

x_train, x_test, y_train, y_test = train_test_split(
     x, y,           # 데이터
     train_size=0.7, #train set 70%
     shuffle=False
)

print(x_train, y_train) 
print(x_test, y_test)  



model = Sequential()
model.add(Dense(5, input_dim=1, activation='linear'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1, validation_split=0.1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

