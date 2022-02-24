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
  
  if 0 < count < idx:  # 각 열에 대한 설명인 첫행은 스킵하기 위해 0 초과, 원하는 데이터 개수의 70퍼센트만큼 train data로
    append_data=line.split(",") # 불러온 행의 요소들을 ',' 단위로 스플릿해준다. (csv 파일이기 때문)
    append_data=list(map(float,append_data)) # 스플릿해준 각 요소들이 스트링 값이라 이를 float로 변환
    
    train_data.append(append_data) # 만들어낸 append data를 train_data에 다시 넣어주면 이중리스트 형성(매 루프마다 리스트가 한개씩 들어가는 꼴)
                                   # 이렇게 만들어낸 train data로 선형회귀 모델을 만들 것이다
  
  if count==idx+1:  # 원하는 만큼 뽑아냈으면 break
    break
    
  count+=1  # 행 세주기
f.close()
count-=1

f_data=[]
target=[]
val_data=[]
val_target=[]

f_len= int((count-1)*0.7)
val_len = int((count-1)*0.3)

# train_data
for i in range(f_len):  # 아까 구한 count 를 활용하여 뽑아낸 행의 개수를 구한다, model train 할때 쓰이는 데이터는 총 데이터의 70프로
  append_target=[]
  #print(train_data[i][3])
  append_target.append(train_data[i][3])  # target data는 더 복잡하게 구하는 이유는 그냥 append 해버리면 이중리스트가 되지 않는다
  #print(append_target)
  target.append(append_target)
  f_data.append(train_data[i][4:])

# valid_data
for i in range(val_len):  # validation data 를 train data에서 30프로 정도를 사용
  append_val_target=[]
  append_val_target.append(train_data[f_len+i][3])
  val_target.append(append_val_target)
  val_data.append(train_data[f_len+i][4:])
    
#tf.random.set_seed(42)
'''
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
'''
model = Sequential([
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1)
])

model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # loss 함수로 categorical crossentropy 를 사용해보니 validation이 일정하게 나온다 -> mse 사용

from tensorflow.python.keras.callbacks import EarlyStopping # EarlyStopping 을 통해 모델 학습에 epoch 20번 동안 진전이 없으면 종료한다
early_stopping = EarlyStopping(patience=20)
model.fit(f_data,target,epochs=1000, batch_size=32,validation_data=(val_data,val_target),callbacks=[early_stopping])

from tensorflow.python.keras.models import load_model   # load model 로 방금 만든 모델을 kaggle output 파일에 저장한다
model.save('ubiquant_model1.h5')
model_1 = load_model('ubiquant_model1.h5')  # 저장한 모델을 다음과 같이 불러온다

import ubiquant
env = ubiquant.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test set and sample submission

for (test_df, sample_prediction_df) in iter_test:
    numpy_test_df=test_df.to_numpy()
    sample_prediction_df['target'] = model_1.predict(numpy_test_df[:,2:].astype(float))  # make your predictions here
    env.predict(sample_prediction_df)   # register your predictions
