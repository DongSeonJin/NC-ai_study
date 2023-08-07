import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import datetime

# 1. 데이터
path = './data/boston/'
x_train = pd.read_csv(path + 'train-date.csv',
                    index_col=0)
x_test = pd.read_csv(path + 'train-date.csv',
                    index_col=0)
y_train = pd.read_csv(path + 'train-date.csv',
                    index_col=0)
y_test = pd.read_csv(path + 'train-date.csv',
                    index_col=0)



# 2. 모델구성

# 3. 컴파일훈련

# 4. 평가, 예측

