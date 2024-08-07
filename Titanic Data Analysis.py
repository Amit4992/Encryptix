#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


import sklearn
sklearn.__version__


# In[3]:


titanic_data=pd.read_csv("C:\\Users\\Amit\\Desktop\\Desktop\\Encrptix Projects\\Titanic-Dataset (1).csv")


# In[4]:


# printing the first five row
titanic_data.head()


# In[5]:


# Numbers of Rows and Columns
titanic_data.shape


# In[6]:


titanic_data.info()


# In[7]:


# checking number of missing values in each column
titanic_data.isnull().sum()


# In[8]:


# Handling the missing values 
# Drop column name Cabin from data frame
titanic_data=titanic_data.drop(columns = "Cabin",axis=1)


# In[9]:


titanic_data.isnull().sum()


# In[10]:


#Replacing null values in age  by mean of age
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)


# In[11]:


# Checking missing values in titanic_data
titanic_data.isnull().sum()


# In[12]:


# checking mode in Embarked column
print(titanic_data["Embarked"].mode())


# In[13]:


# Replacing Two missing values in Embarked by it's mode
titanic_data["Embarked"].fillna(titanic_data["Embarked"].mode()[0],inplace=True)


# In[14]:


# checking missing values in Embarked Column
titanic_data.isnull().sum()


# In[15]:


# Data Analysis
titanic_data.describe()


# In[16]:


# finding counts of numbers of people who survived
titanic_data["Survived"].value_counts()


# In[17]:


# Data Visualization
sns.set()


# In[18]:


titanic_data.head()


# In[19]:


# making count for sex column
sns.countplot(x='Sex', data=titanic_data)
plt.show()


# In[20]:


# Number of survivor gender wise
sns.countplot(x="Sex",hue="Survived",data=titanic_data)
plt.show()


# In[21]:


# Making count plot for Pclass
sns.countplot(x='Pclass',data=titanic_data)
plt.show()


# In[22]:


# count of Pclass based on number of people survived
sns.countplot(x='Pclass',hue="Survived",data=titanic_data)
plt.show()


# In[23]:


# Making a countplot on the basis of embarked column
sns.countplot(x="Embarked",data=titanic_data)
plt.show()


# In[24]:


sns.countplot(x="Embarked",hue="Survived",data=titanic_data)
plt.show()


# In[25]:


# Encoding the categorical columns for Sex column
titanic_data["Sex"].value_counts()


# In[26]:


# Replacing categorical values to Numerical
titanic_data=titanic_data.replace({"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}})


# In[81]:


titanic_data.head()


# In[27]:


titanic_data.drop(columns=["PassengerId","Name","Survived","Ticket"],axis=1)


# In[28]:


x=titanic_data.drop(columns=["PassengerId","Name","Ticket","Survived"],axis=1)
print(x)


# In[29]:


y=titanic_data["Survived"]
print(y)


# In[30]:


# Spliting the data into training data and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)


# In[31]:


# printing shape of x,x_train, x_test
print(x.shape,x_train.shape,x_test.shape)


# In[32]:


# Model Training
Model=LogisticRegression()


# In[33]:


# Training the logistic Regression Model with training data
Model.fit(x_train,y_train)


# In[34]:


x_train_prediction=Model.predict(x_train)
print(x_train_prediction)


# In[35]:


training_data_accuracy=accuracy_score(y_train,x_train_prediction)
print('Accuracy Score of Training Data:',training_data_accuracy)


# In[36]:


x_test_prediction=Model.predict(x_test)
print(x_test_prediction)


# In[37]:


test_data_accuracy=accuracy_score(y_test,x_test_prediction)
print("Accuracy score of Test Data:",test_data_accuracy)

