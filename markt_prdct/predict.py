import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.linear_model import LinearRegression

# visulize 할 index 정하기
idx=4
idx_2=6

# csv 파일을 리스트로 변환해주기
## train data
f=open("/content/train.csv","r") # 디렉토리는 알아서 설정

reader=csv.reader(f)
count=0
train_data=[]
train_data_2=[]

for row in reader:
  if count == idx :
    train_data.append(row)
  elif count == idx_2:
    train_data_2.append(row)
    break
  else:
    count+=1
f.close()

# Linear Regression
plt_dt=train_data[0][4:]
plt_data=np.array(plt_dt).reshape((1,-1))
plt_dt_2=train_data_2[0][4:]
plt_data_2=np.array(plt_dt_2).reshape((1,-1))

target=(np.array(train_data[0][3])).reshape((-1,1))
target_2=(np.array(train_data_2[0][3])).reshape((-1,1))

t=range(len(plt_dt))

reg=LinearRegression()
reg.fit(,plt_data)
coef=reg.coef_
intercept=reg.intercept_

## Build the linear function
def f(x):
  return float(coef)*x + float(intercept)

#plot
plt.plot(t,plt_dt,'b')
plt.plot(t,f(t),'r')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.show()
