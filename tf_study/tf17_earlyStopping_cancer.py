import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import RobustScaler
from keras.callbacks import EarlyStopping

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape)  # (569, 30)
print(y.shape)  # (569,)
print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=1234,
    shuffle=True
)

print(x_train.shape, x_test.shape) # (398, 30)
print(y_train.shape, y_test.shape)

# 2. 모델구성
model = Sequential()
model.add(Dense(68, input_dim=30))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1, activation='sigmoid')) # 이진분류는 마지막 아웃풋 레이어에 무조건
                                            #  sigmoid 함수 사용!!!!

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['mse', 'accuracy'])

early_stopping = EarlyStopping(monitor='val_loss',
                                patience=50,
                                mode='min',
                                verbose= 1,
                                restore_best_weights=True)

model.fit(x_train, y_train, epochs=100,
          batch_size=128,
          callbacks=[early_stopping])

# 4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mse : ', mse)
print('accuracy : ', accuracy)

######## 실습 : accuracy_score 를 출력하기!!!! ########
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
# print(y_predict)

########### 코드 한 줄 필요 #########
act_score = accuracy_score(y_test, y_predict)
print('acc_score : ', act_score) 

