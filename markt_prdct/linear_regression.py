import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf

# visulize 할 index 정하기
idx=100 # 무조건 10의 배수!!!!!!!!!!!!


# csv파일의 데이터 값들을 반복문으로 이중리스트 만들어주기
f = open("/content/train.csv",'r')
count=0
train_data=[]
test_data=[]

while True:
  line=f.readline() # 먼저, 오픈한 f를 한줄한줄 읽어온다
  
  if 0 < count < idx*0.7+1:  # 각 열에 대한 설명인 첫행은 스킵하기 위해 0 초과, 원하는 데이터 개수의 70퍼센트만큼 train data로
    append_data=line.split(",") # 불러온 행의 요소들을 ',' 단위로 스플릿해준다. (csv 파일이기 때문)
    append_data=list(map(float,append_data)) # 스플릿해준 각 요소들이 스트링 값이라 이를 float로 변환
    
    train_data.append(append_data) # 만들어낸 append data를 train_data에 다시 넣어주면 이중리스트 형성(매 루프마다 리스트가 한개씩 들어가는 꼴)
                                   # 이렇게 만들어낸 train data로 선형회귀 모델을 만들 것이다
      
  if idx*0.7 < count < idx+1: # 만든 모델을 평가할 test data를 만든다, 전체 데이터의 30퍼센트만큼
    append_data=line.split(",")
    append_data=list(map(float,append_data))

    test_data.append(append_data)
  
  if count==idx+1:  # 원하는 만큼 뽑아냈으면 break
    break

  count+=1  # 행 세주기
f.close()
count-=1 # 각 열들을 나타내주는 0행을 제외하고 넣어줬기 때문에 다시 빼줘야 얻어낸 총 데이터 개수이다


# train data 의 f data, target 값들 모아둔 리스트 각각 만들기
f_data=[]
target=[]

for i in range(int(count*0.7)):  # 아까 구한 count 를 활용하여 뽑아낸 행의 개수를 구한다
  append_target=[]
  append_target.append(train_data[i][3])  # target data는 더 복잡하게 구하는 이유는 그냥 append 해버리면 이중리스트가 되지 않는다
  target.append(append_target)
  f_data.append(train_data[i][4:])

  
# 똑같이 test data 다듬기
test_fdata=[]
test_target=[]

for i in range(int(count*0.3)):
  test_fdata.append(test_data[i][4:])
  test_target.append(test_data[i][3])  # test data의 target은 선형회귀를 하는게 아니라 나중에 평가용이므로 이중리스트로 만들지 않는다
  
  
# Linear Regression
reg=LinearRegression()
reg.fit(f_data,target)


# Linear Regression Predict
predict=reg.predict(test_fdata)
predict=np.array(predict).reshape(-1)

#keras linear Regression model Predict

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, activation='linear'))
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0) 
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(f_data, target, batch_size=64, epochs = 100,  shuffle=True)
model.predict(f_data)
model.evaluate(test_fdata, test_target)

predict_k =model.predict(test_fdata)
predict_k =np.array(predict_k).reshape(-1)
evaluate = model.evaluate(test_fdata, test_target)


#linear regression plot

t=list(range(int(count*0.3)))
plt.plot(t,test_target,'r',label='target')
plt.plot(t,predict,'b',label='predicted')
plt.xlabel('Data')
plt.ylabel('Price')
plt.legend(loc='best', ncol=2)
plt.show()


#keras linear model plot

t=list(range(int(count*0.3)))
plt.plot(t,test_target,'r',label='target')
plt.plot(t,predict_k,'b',label='predicted')
plt.xlabel('Data')
plt.ylabel('Price')
plt.legend(loc='best', ncol=2)
plt.show()

print(evaluate) 
