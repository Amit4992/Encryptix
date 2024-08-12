#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import numpy for numerical calculation.
import numpy as np
# import pandas for data analysis.
import pandas as pd
# import matplotlib.pyplot ans seaborn for data visualization.
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.linear_model import logisticRegression for supervised learning algorithum used for binary classification task.
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split that splits the data set into two subsets training set for model fitting
# testing set for model evaluation.
from sklearn.model_selection import train_test_split
# from sklearn.metrics import acccuracy_score for measuring models accuracy, higher accuracy means model is good.
from sklearn.metrics import accuracy_score
# import plotly for interactive data visualization
import plotly.express as px
import plotly.graph_objects as go
# import scipy
import scipy
# import libraries
from scipy.stats import skew
from scipy.stats import kurtosis
import pylab as p
import statsmodels.api as sm


# In[3]:


Sales=pd.read_csv("C:\\Users\\Amit\\Downloads\\advertising.csv")


# In[6]:


Sales.head()


# In[7]:


Sales.describe()


# In[8]:


Sales.isnull()


# In[9]:


Sales.info()


# In[14]:


# Outliers Analysis
fig, axs = plt.subplots(3, figsize=(5,5))
plt1 = sns.boxplot(Sales['TV'],ax=axs[0])
plt2 = sns.boxplot(Sales['Radio'],ax=axs[1])
plt3 = sns.boxplot(Sales['Newspaper'],ax=axs[2])
plt.tight_layout()


# In[4]:


# lets see how sales are related to other variables using scatter plot
sns.pairplot(Sales,x_vars=["TV","Newspaper","Radio"],y_vars="Sales",height=4,aspect=1,kind="scatter")
plt.show()


# In[6]:


# let's see the correlation between different variables
plt.figure(figsize=(5,5))
sns.heatmap(Sales.corr(),cmap="YlGnBu",annot=True,fmt=".2f")
plt.show()


# In[7]:


x=Sales["TV"]
y=Sales["Sales"]


# In[9]:


print(x)
print(y)


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,test_size=0.3,random_state=2)


# In[21]:


print("Count of x_train :", x_train.count())
print("Count of y_train :", x_train.count())


# In[23]:


x_train.head()


# In[24]:


y_train.head()


# In[26]:


# Add a constant to get a intercept
x_train_sm=sm.add_constant(x_train)
# Fit the regression line using 'OLS'
lr=sm.OLS(y_train,x_train_sm).fit()


# In[27]:


lr.params


# In[29]:


print(lr.summary())


# In[30]:


plt.scatter(x_train,y_train)
plt.plot(x_train,6.6755+0.0577*x_train,'r')


# In[ ]:




