from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import time
import datetime

# 1. 데이터
train_dategen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    width_shift_range = 0.1,
    rotation_range = 5,
    zoom_range = 1.2,
    shear_range = 0.5,
    fill_mode = 'nearest',
    validation_split = 0.2
)
test_datagen = ImageDataGenerator(
    rescale = 1./255
)

xy_train = test_datagen.flow_from_directory(
    '/home/ncp/workspace/_data/rps/',
    target_size = (150, 150),
    batch_size = 125,
    class_mode = 'categorical',
    color_mode = 'rgb',
    shuffle = True,
    subset = 'training'
)

print(xy_train[1][0].shape)
print(xy_train[0][1].shape)

xy_test = test_datagen.flow_from_directory(
    '/home/ncp/workspace/_data/rps/',
    target_size = (150, 150),
    batch_size = 128,
    class_mode = 'categorical',
    color_mode = 'rgb',
    shuffle = True,
    subset = 'validation'
)

print(xy_train[0][0].shape)
print(xy_train[0][1].shape)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape=(150,150,3),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', #데이터는 원핫인코딩 되어있는상태임
             optimizer='adam',
             metrics=['accuracy'])
earlyStopping = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience = 20,
    restore_best_weights=True,
    verbose=1
)

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filepath = '/home/ncp/workspace/_mcp/'
filename = '{epoch:04d}-{val_loss: .4f}.hdf5'

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath="".join([filepath, 'rps', date, '_', filename])
)

start_time = time.time()

model.fit(xy_train[0][0], xy_train[0][1],
         validation_split=0.2,
          epochs=30, batch_size=128,
          verbose=1,
          callbacks=[earlyStopping, mcp]
         )

end_time = time.time() - start_time

# 4. 평가, 예측
loss, acc = model.evaluate(xy_test[1][0], xy_test[0][1])

print('loss : ', loss)
print('acc : ', acc)
print('걸린 시간 : ', end_time)