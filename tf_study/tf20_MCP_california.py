import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

########### 실습 : r2 score를 0.6으로 높이기!!!!! ###########
# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape) # (20640, 8)
print(y.shape) # (20640,)
print(datasets.feature_names)
print(datasets.DESCR)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,
    random_state=1234,
    shuffle=True
)
print(x_train.shape) #(14447, 8)
print(y_train.shape) #(14447,)
print(x_test.shape)  #(6193, 8)
print(y_train.shape) #(14447,)

#2.모델구성
model = Sequential()
model.add(Dense(128, input_dim=8, activation='linear'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='rmsprop')
early_stopping = EarlyStopping(monitor='val_loss',
                                patience=50,
                                mode='min',
                                verbose= 1,
                                restore_best_weights=True)

#  Model check point
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./_mcp/tf20_california.hdf5'
)
model.fit(x_train, y_train, epochs=300, 
          batch_size=64, validation_split=0.2,
            callbacks=[early_stopping])


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

###### r2 score 결정계수 ######
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)