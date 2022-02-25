import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

idx=10000
f = open("/kaggle/input/ubiquant-market-prediction/train.csv",'r')
count=0

train_data=[]

while True:
  line=f.readline() # 먼저, 오픈한 f를 한줄한줄 읽어온다

    
  if idx < count < 2*idx+1:  # 각 열에 대한 설명인 첫행은 스킵하기 위해 0 초과, 원하는 데이터 개수의 70퍼센트만큼 train data로
    append_data=line.split(",") # 불러온 행의 요소들을 ',' 단위로 스플릿해준다. (csv 파일이기 때문)
    append_data=list(map(float,append_data)) # 스플릿해준 각 요소들이 스트링 값이라 이를 float로 변환
    
    train_data.append(append_data) # 만들어낸 append data를 train_data에 다시 넣어주면 이중리스트 형성(매 루프마다 리스트가 한개씩 들어가는 꼴)
                                   # 이렇게 만들어낸 train data로 선형회귀 모델을 만들 것이다

  if count==2*idx+1:  # 원하는 만큼 뽑아냈으면 break
    break
    
  count+=1  # 행 세주기
f.close()

#K-fold

print(len(train_data))
f_data=[]
target=[]
val_data=[]
val_target=[]

f_len= int(idx*0.7)
val_len = int(idx*0.3)
#print(f_len)
#print(val_len)

# train_data
for i in range(f_len):  # 아까 구한 count 를 활용하여 뽑아낸 행의 개수를 구한다
  append_target=[]
  #print(train_data[i][3])
  append_target.append(train_data[i][3])  # target data는 더 복잡하게 구하는 이유는 그냥 append 해버리면 이중리스트가 되지 않는다
  #print(append_target)
  target.append(append_target)
  f_data.append(train_data[i][4:])
i+=1
#print(f'i={i}')    

# valid_data
for j in range(val_len):
  append_val_target=[]
  append_val_target.append(train_data[i+j][3])
  val_target.append(append_val_target)
  val_data.append(train_data[i+j][4:])
    
#tf.random.set_seed(42)
'''
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
'''
model = Sequential([
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dense(1)
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit(f_data,target,epochs=1000, batch_size=64,validation_data=(val_data,val_target),callbacks=[early_stopping])
