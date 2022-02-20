import matplotlib.pyplot as plt
import csv
import numpy as np
import keras
from tensorflow import keras
from tensorflow.keras import layers

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

for i in range(count-1):  # 아까 구한 count 를 활용하여 뽑아낸 행의 개수를 구한다
  append_target=[]
  
  
  #print(train_data[i][3])
  append_target.append(train_data[i][3])  # target data는 더 복잡하게 구하는 이유는 그냥 append 해버리면 이중리스트가 되지 않는다
  #print(append_target)
  target.append(append_target)
  f_data.append(train_data[i][4:])

# 간단한 linear regression model(layer 1개)    
'''
model = keras.Sequential()
model.add(layers.Dense(1,activation='linear'))
optimizer= keras.optimizers.SGD(learning_rate=0.001, momentum=0.0)
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(f_data, target, batch_size=64, epochs = 100,  shuffle=True)
'''

# layer 3개, relu regression. 점수: 0.034

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

#tf.random.set_seed(42) # 파라미터를 생성하는 방식 고정

model = Sequential([
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1)
])

model.compile(loss=rmse, optimizer=Adam(), metrics=[accuracy])  # metrics rmse 보다는 accuracy가 조금더 정확도 높았음 (score: 0.076)

model.fit(f_data, target, epochs=100)

# 제출 및 점수 
import ubiquant
env = ubiquant.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test set and sample submission

for (test_df, sample_prediction_df) in iter_test:
    numpy_test_df=test_df.to_numpy()
    sample_prediction_df['target'] = model.predict(numpy_test_df[:,2:].astype(float))  # make your predictions here
    env.predict(sample_prediction_df)   # register your predictions
