import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17 ,18 ,19, 20 ])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17 ,18 ,19, 20 ])
x = np.array

# print(x.shape) #(20,)
# print(y.shape) #(20,)

x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     11, 12])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     11, 12])

x_val = np.array([13, 14, 15, 16])
y_val = np.array([13, 14, 15, 16])

x_test = np.array([17, 18, 19, 20])
y_test = np.array([17, 18, 19, 20])

# 2. 모델구성
model = Sequential()
model.add(Dense(14, input_dim=1))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_data=[x_val, y_val])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([21])
print('21의 예측값 : ', result)