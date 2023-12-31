import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (150, 4) (150,)
print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)',]
print(datasets.DESCR)

#### 원핫인코딩 one-hot encoding #####
from keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=72,
    shuffle=True
)

print(x_train.shape, x_test.shape)  # (105, 4) (45, 4) => 4는 인풋
print(y_train.shape, y_test.shape)  # (105, 3) (45, 3) => 3은 아웃풋

scaler = StandardScaler()
x = scaler.fit_transform(x)

# 2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=4))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['mse', 'accuracy']) # 회귀분석은 mse, r2 score
                                           # 분류분석은 mse, accuracy score

early_stopping = EarlyStopping(monitor='val_loss',
                                patience=50,
                                mode='min',
                                verbose= 1,
                                restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./_mcp/tf20_iris.hdf5'
)


start_time = time.time()

model.fit(x_train, y_train,
          epochs=500,
          batch_size=32,
          callbacks=[early_stopping])

end_time = time.time() - start_time
print('걸린시간 : ', end_time)

# 4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mse : ', mse)
print('accuracy : ', accuracy)