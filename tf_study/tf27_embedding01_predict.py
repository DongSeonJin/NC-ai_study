import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

# 1. 데이터
docs = ['재미있어요', '재미없다', '돈 아깝다', '최고에요', '배우가 잘 생겼어요', '추천해요', '글쎄요', '감동이다', '최악', '후회된다', '보다 나��다', '발연기에요', '꼭봐라', '세 번 봐라', '또보고싶다', '돈버렸다', '다른 거 볼걸', 'n회차 관람', '다음편 나왔으면 좋겠다', '연기가 어색해요', '줄거리가 이상해요', '숙면했어요', '망작이다', '차라리 집에서 잘걸', '즐거운 시간보냈어요']

# 긍정 1, 부정 0

labels = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1])

# Tokenizer
token = Tokenizer()
token.fit_on_texts(docs) # index화

x = token.texts_to_sequences(docs)
print(x)

# pad_sequencse

pad_x = pad_sequences(x, padding = 'pre', maxlen = 5)
print(pad_x)
print(pad_x.shape) 

# word_size 
word_size = len(token.word_index)
print('word_size : ', word_size)

# 2. 모델구성
model = Sequential()
model.add(Embedding(input_dim = word_size + 1, output_dim = 32, input_length = 5))
model.add(LSTM(32)) 
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

# 3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(pad_x, labels, epochs= 100, batch_size= 32)

# 4. 평가, 예측
loss, acc = model.evaluate(pad_x, labels)
print('loss : ', loss)
print('acc : ', acc)

# predice # 

predict1 = '정말 재미있고 최고였어요'
predict2 = '진짜 후회된다 최악'

# 1) tokenizer
token = Tokenizer()
x_predict = np.array([predict2])

token.fit_on_texts(x_predict) # index화 
token.word_index
print(token.word_index) # {'진짜': 1, '후회된다': 2, '최악': 3}

x_pred = token.texts_to_sequences(x_predict) # 문장 정수화
print(x_pred)

# 2) pad_sequences
x_pred = pad_sequences(x_pred, padding = 'pre', maxlen = 5)
print(x_pred)

# 3) predict
y_pred = model.predict(x_pred)
print(y_pred)

# 1. predict 값이 잘 나올 수 있도록 모델구성과 데이터(docs) 수정
# 2. 결과 값을 '긍정' 과 '부정'으로 출력하시오

score = float(model.predict(x_pred))

if y_pred > 0.5:
    print('{:.2f}%의 확률로 부정'.format((1-score)*100))
else:
    print('{:.2f}%의 확률로 긍정'.format((score)*100))  
