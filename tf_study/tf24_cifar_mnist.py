from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Dropout, Flatten, Conv2D
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

# 1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)      # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)        # (10000, 32, 32, 3) (10000, 1)

# 시각화

# plt.imshow(x_train[5])
# plt.show()

# Scaling

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

# 2. 모델구성
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 activation='relu',
                 padding='same',
                 input_shape=(32, 32, 3)))
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(13, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

earlyStopping = EarlyStopping(
    mode='min',
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()

model.fit(
    x_train, y_train, validation_split=0.2,
callbacks=[earlyStopping],
    epochs=100, batch_size=32
)

end_time = time.time() - start_time

# 4. 평가, 예측

loss, acc =model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc :', acc)
print('걸린 시간 :', end_time)


# 최종