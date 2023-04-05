# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
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
df.head()

![df head](https://user-images.githubusercontent.com/118707009/230002983-032b365c-10c6-4a0f-a827-23ef50f10edb.png)

df.tail()

![df tail](https://user-images.githubusercontent.com/118707009/230003054-5fc312d0-28e9-4e4d-967d-f45dbb126c08.png)

Array Value of X

![array value of x](https://user-images.githubusercontent.com/118707009/230003161-3e0aea3a-1b8b-449b-91bf-2b12dd1068c2.png)

Array Value of Y

![array value of y](https://user-images.githubusercontent.com/118707009/230003211-2ef9df33-b190-4c89-8601-8617e0ca6a6a.png)

Values of Y Prediction

![values of y prediction](https://user-images.githubusercontent.com/118707009/230003343-f5f9ce3c-84bf-48c1-85f9-b7f05987c325.png)

Array Values of Y test

![array values of y test](https://user-images.githubusercontent.com/118707009/230003492-7e70cb0f-7c83-402b-ae1e-0c5911dc9a3b.png)

Training Set Graph

![training set graph](https://user-images.githubusercontent.com/118707009/230004169-6f5db1c6-67c7-4c5d-8207-dce620f42905.png)

Test Set graph

![test set graph](https://user-images.githubusercontent.com/118707009/230003667-8dc99f0d-e844-4889-a1fd-64035da8a2d8.png)

Values of MSE, MAE and RMSE

![values of mse,mae,rmse](https://user-images.githubusercontent.com/118707009/230003876-de24b094-d8e1-401e-8925-0443c6474bad.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
