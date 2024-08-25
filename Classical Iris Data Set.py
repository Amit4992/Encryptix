#!/usr/bin/env python
# coding: utf-8

# In[1]:


# standard imports
import os
#Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
#local imports


# In[2]:


sns.set()


# # Load Data 
#  
#      let's load the iris flower dataset using scikit-learn's build-in datasets.

# In[3]:


data=datasets.load_iris()


# In[4]:


data.keys()


# In[5]:


print(data["DESCR"])


# In[6]:


data["data"][:5]


# In[7]:


data['feature_names']


# In[8]:


data['target']


# In[9]:


data["target_names"]


# # What problem are we going to solve?
# we are trying to use attributes of flowers to predict the species of the flower.Specifically,we are trying to use sepal length and width and petal length and petal width tp predict if an iris flower is of type Setosa,Versicolor, or Virginica.
# This is a multiclass classification problem

# # Create a pandas DataFrame From data 
# We could do our full analysis using Numpy and Numpy arrays, but we'll create a pandas DataFrame because it does makes some things simplier

# In[10]:


df = pd.DataFrame(data["data"], columns=data["feature_names"])


# In[11]:


df["target"]=data["target"]


# In[12]:


df.head()


# # Basic Statistics

# In[13]:


df.describe()


# In[14]:


col="sepal length (cm)"
df[col].hist()
plt.xlab()
plt.suptitle(col)
plt.show()


# In[15]:


col="sepal width (cm)"
df[col].hist()
plt.suptitle(col)
plt.show()


# In[16]:


col="petal length (cm)"
df[col].hist() 
plt.suptitle(col)
plt.show()


# In[17]:


col="petal width (cm)"
df[col].hist()
plt.suptitle(col)
plt.show()


# # Relationship with data feature with targets

# In[18]:


col= "sepal length (cm)"
sns.relplot(x=col,y="target",hue="target",data=df)


# In[19]:


df.head()


# In[20]:


data["target_names"]


# In[21]:


# mapping of data set
df["target_name"] = df["target"].map({0:"setosa",1:"versicolor",2:"virginica"})


# In[26]:


df.head()


# In[22]:


col="sepal length (cm)"
sns.relplot(x=col,y="target",hue="target_name",data=df)
plt.suptitle(col,y=1.05)
plt.show()


# # Explotary Data Analysis (EDA) - Pairplots

# In[23]:


sns.pairplot(df)


# In[24]:


df.pop("target_name")


# In[25]:


sns.heatmap(df)


# In[27]:


df.head()


# In[36]:


sns.pairplot(data=df,hue="target_name")


# # Train Test Split
# 

# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


get_ipython().run_line_magic('pinfo', 'train_test_split')


# In[57]:


df_train,df_test=train_test_split(df,test_size=0.25)


# In[58]:


df_train.shape


# In[59]:


df.head()


# In[60]:


df["target_name"]=df["target"].map({0:"setosa",1:"versicolor",2:"virginica"})


# In[61]:


df.head()


# In[62]:


df_train,df_test=train_test_split(df,test_size=0.25)


# In[63]:


df_train.shape


# In[64]:


df_test.shape


# In[65]:


df_train.head()


# In[66]:


df_test.head()


# # Data for Modeling
# this involves splitting the data back into plain  NumPy arrays.
# 

# In[67]:


x_train=df_train.drop(columns=["target","target_name"]).values
x_train


# In[68]:


x_train.shape


# In[69]:


y_train=df_train["target"].values


# In[70]:


y_train


# In[71]:


y_train.shape


# # Modeling- What is our Baseline?
# what is the simplest model we can think of?
# 
# In this case,if our baseline model is randomly guessing the species of flower, or guessing a single species for every data point ,we would expect to have a model accuracy of 0.33 or 33%,since we have three diffrent that are evenly balanced.
# 
# so our model should be atlest 33% accuracy.
# 
# # Modeling - Simple manual model
# 
# Let's manually look our data and decide some cutoff points for classification.
# 
# 

# In[72]:


data["target_names"]


# In[73]:


def single_feature_prediction(petal_length):
    """Predicts the Iris species given the petal length."""
    if petal_length<2.5:
        return 0
    elif petal_length<4.5:
        return 1 
    else:
        return 2


# In[74]:


x_train[:,2]


# In[75]:


manual_y_Prediction= np.array([single_feature_prediction(val) for val in x_train[:,2]])


# In[76]:


manual_y_Prediction


# In[77]:


np.mean(manual_y_Prediction==y_train)


# In[ ]:





# In[ ]:




