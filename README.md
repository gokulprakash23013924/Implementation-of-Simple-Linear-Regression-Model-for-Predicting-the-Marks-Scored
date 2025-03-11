# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
### gokulprakash m(212223240041)
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
### Dataset
![Screenshot 2025-03-11 133838](https://github.com/user-attachments/assets/2a0480e8-0a95-4208-8f43-d1a6d2cfd480)

### Head Values
![Screenshot 2025-03-11 133850](https://github.com/user-attachments/assets/24867ad6-ca28-41f2-adad-c4e7936b7969)

### Tail Values
![Screenshot 2025-03-11 133857](https://github.com/user-attachments/assets/3b57ef67-fc0a-42a0-902a-23e3cb0c9cd3)

### X and Y values
![Screenshot 2025-03-11 133908](https://github.com/user-attachments/assets/b2287ca8-64e0-45ab-b530-c168a2806457)

### Predication values of X and Y
![Screenshot 2025-03-11 134055](https://github.com/user-attachments/assets/bd3a6635-b13e-4d6b-9909-ed026dcbe695)

### MSE,MAE and RMSE
![Screenshot 2025-03-11 134141](https://github.com/user-attachments/assets/7fd112ba-964a-45c5-8990-025259a0fc7a)

### Training Set
![Screenshot 2025-03-11 134117](https://github.com/user-attachments/assets/db577bee-e112-4aaf-bdbb-cdc844be7b6e)

### Testing Set
![Screenshot 2025-03-11 134134](https://github.com/user-attachments/assets/529c0a88-8c7b-457f-adf7-7e20c66dfd7c)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
