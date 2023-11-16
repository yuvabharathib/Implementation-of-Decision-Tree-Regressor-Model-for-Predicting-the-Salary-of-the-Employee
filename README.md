# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
  
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import dataset and get data info
   
2. check for null values
   
3. Map values for position column
   
4. Split the dataset into train and test set
   
5. Import decision tree regressor and fit it for data
    
6. Calculate MSE,R2 and y predict.

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Yuvabharathi.B
RegisterNumber:  212222230181


import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
l0=LabelEncoder()

data["Position"]=l0.fit_transform(data['Position'])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:

data.head()

![41](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/111515488/919484c4-ae07-4355-aa0f-24a5682ac59c)


data.info()

![42](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/111515488/f5bad7e0-46c5-4dc2-9926-9b21e2cee7dd)


isnull() and sum()

![43](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/111515488/09ff361e-8ea5-4863-820d-55905532b50b)


data.head() for salary

![44](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/111515488/7c49a096-ada0-42be-a306-90954f7e9549)


MSE Value

![45](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/111515488/37a9f831-3d46-4f02-bdee-dfd35bdc9cb2)

r2 value 

![46](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/111515488/c93c0522-fd84-49f8-80b1-66d4d124e51d)


data prediction

![47](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/111515488/c7fe8a00-4430-4873-99aa-244281a2ec1b)


Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
