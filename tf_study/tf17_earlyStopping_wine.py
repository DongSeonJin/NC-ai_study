import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
import time

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(178, 13) (178,)
print(datasets.feature_names)
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
#   'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols'
#   , 'proanthocyanins', 'color_intensity', 'hue',
#     'od280/od315_of_diluted_wines', 'proline']
print(datasets.DESCR)
# class:             - class_0
#                    - class_1
#                    - class_2
# one_hot_encoding
y = to_categorical(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=64,
    shuffle=True
)

print(x_train.shape, x_test.shape)  #(124, 13) (54, 13)
print(y_train.shape, y_test.shape)  #(124, 3) (54, 3)

scaler = StandardScaler()
x = scaler.fit_transform(x)

# 2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['mse', 'accuracy'])

early_stopping = EarlyStopping(monitor='val_loss',
                                patience=50,
                                mode='min',
                                verbose= 1,
                                restore_best_weights=True)

start_time = time.time()

model.fit(x_train, y_train, epochs=500, batch_size=64,
          callbacks=[early_stopping])

end_time = time.time() - start_time
print('걸린시간 : ', end_time)

#4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mse : ', mse)
print('accuracy : ', accuracy)
print('걸린시간 : ', end_time)