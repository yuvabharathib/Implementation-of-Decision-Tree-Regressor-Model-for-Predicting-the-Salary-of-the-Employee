## Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
import dataset and get data info
check for null values
Map values for position column
Split the dataset into train and test set
Import decision tree regressor and fit it for data
Calculate MSE,R2 and y predict.
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
### data.head()
![image](https://github.com/yuvabharathib/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113497404/f3bf1859-e235-4ed7-9aad-f8848f3b7c52)

### data.info()
![image](https://github.com/yuvabharathib/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113497404/0c02414d-a715-46d8-9c6b-fc69fa1fc152)

### isnull() and sum()
![image](https://github.com/yuvabharathib/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113497404/f43475c3-d7a8-41d4-8475-b0b0f6cadbcc)

### data.head() for salary
![image](https://github.com/yuvabharathib/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113497404/9383022d-84c0-41b5-80a2-a909ccbc40f2)

### MSE Value
![image](https://github.com/yuvabharathib/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113497404/50b5d1d9-822d-4ab6-a192-6c6a21ca0794)

### r2 value
![image](https://github.com/yuvabharathib/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113497404/72be5a49-990d-492c-8800-c2d292086725)

### data prediction
![image](https://github.com/yuvabharathib/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113497404/dac99ee8-27c4-4e7a-af7a-c4f2f5204fa9)


## Result: 
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming
