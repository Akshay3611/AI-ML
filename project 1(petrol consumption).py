# -*- coding: utf-8 -*-
"""
@author: Akshay

This is a demo project on petrol consumption
"""
# Project 1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Importing the client data and checking records via Variable explorer
data=pd.read_csv("C:/Users/STUDENT/Desktop/GitHub Repo/petrol_consumption.csv")
# Checking rows and columns in Dataset
data.shape
# Checking Null values in dataset
data.isnull().sum()
#As there are no null value present in dataset

# Finding Mean of dataset
import statistics
mean=statistics.mean(data.Petrol_tax)
print("Mean of Petrol tax is : ",mean)
mean=statistics.mean(data.Average_income)
print("Mean of Average income is : ",mean)
mean=statistics.mean(data.Paved_Highways)
print("Mean of Paved Highways is : ",mean)

# Renaming column name 

data.rename(columns={"Population_Driver_licence(%)":"Population_Driver_licence"},inplace=True) 
data
mean=statistics.mean(data.Population_Driver_licence)
print("Mean of Population Driver licence(%) with is : ",mean)
mean=statistics.mean(data.Petrol_Consumption)
print("Mean of Petrol consumption is : ",mean)
data.describe()
# Creating boxplot 
plt.boxplot(data)


# Checking correlation Between variables
data.corr()


sns.heatmap(data.corr(),annot=True)

# Creating Boxplot for column
sns.boxplot(data["Population_Driver_licence"])


# Creating a graph
plt.scatter(data.Population_Driver_licence,data.Petrol_Consumption)

# Applying LinearRegression 
from sklearn import linear_model
model=linear_model.LinearRegression()
model.fit(data[["Population_Driver_licence"]],data.Petrol_Consumption)
model.predict([[0.356]])
plt.xlabel("Population_Driver_licence")
plt.ylabel("Petrol_Consumption")
plt.scatter(data.Population_Driver_licence,data.Petrol_Consumption,color="red",)
plt.plot(data.Population_Driver_licence,model.predict(data[["Population_Driver_licence"]]))
# using train test split Method to Train Data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data[['Petrol_Consumption']],data[['Population_Driver_licence']],train_size=0.7)

X_test
X_train
y_train
y_test
y_predicted=model.predict(X_test)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test,y_predicted,labels = [1,0])
print("confusion matrix : \n",matrix)









