

# 1. 데이터
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))  # 입력층
model.add(Dense(5))               # 히든레이어1
model.add(Dense(7))               # 히든레이어2
model.add(Dense(9))               # 히든레이어3
model.add(Dense(3))               # 히든레이어4
model.add(Dense(1))               # 출력층

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([4])
print('result :', result)