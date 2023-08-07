import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras_preprocessing.sequence import pad_sequences
from keras.datasets import imdb

# 1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
# x 데이터는 자연어, y 데이터는 라벨

print(x_train)
print(x_train.shape, y_train.shape)  #(25000,) (25000,)
print(np.unique(y_train, return_counts=True))
# (arry([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
# 0은 부정 12500개, 1은 긍정 12500개

# 최대 길이와 평균 길이
print('x_train 리뷰의 최대 길이 : ', max(len(i) for i in x_train))
print('x_test 리뷰의 평균 길이 : ', sum(map(len, x_train)) / len(x_train))

# pad_sequences
x_train = pad_sequences(x_train,
                        pedding='pre',
                        maxlen=2494,
                        truncating='pre')
x_test = pad_sequences(x_train,
                       padding='pre',
                       maxlen=2494,
                       truncating='pre')
x_test = pad_sequences(x_test,
                       padding='pre',
                       maxlen='2494',
                       truncating='pre')
print(x_train)