#!/usr/bin/env python
# coding: utf-8

# # OASIS INFOBYTE SIP August data science Internship TASK 1
# Iris Flower Classification Using Machine LEARNING.
# 
# 

# The aim is to classify iris flowers among three species (setosa, versicolor, or virginica) from measurements of sepals and petals' length and
# 
# width.
# 
# The iris data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.
# 
# The central goal here is to design a model that makes useful classifications for new flowers or, in other words, one which exhibits good
# 
# generalization.
# 
# The data source is the file iris_flowers.csv. It contains the data for this example in comma-separated values (CSV) format. The number of
# 
# columns is 5, and the number of rows is 150.

# # Steps to build a ML model: 
#   1.import dataset. 
#   2.visualize the dataset. 
#   3.Data prepartion. 
#   4.Training the algorithm. 
#   5.Making Predicton. 
#   6.Model Evolution.

# In[4]:


import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix


import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

print("Libraries Imported.")


# # 1.Importing Dataset

# In[5]:


df=pd.read_csv('C:\\ProgramData\\anaconda3\\Iris.csv')


# In[7]:


df.head()


# In[8]:


df.head(10)


# In[9]:


df.tail()


# In[10]:


df.shape


# In[11]:


df.isnull().sum()


# In[12]:


df.dtypes


# In[42]:


data=df.groupby('Species')


# In[14]:


data.head()


# In[15]:


df['Species'].unique()


# In[16]:


df.info()


# In[17]:


df.describe()


# # 2.visualizating the dataset

# In[18]:


plt.boxplot(df['SepalLengthCm'])


# In[43]:


plt.boxplot(df['SepalWidthCm'])


# In[45]:


plt.boxplot(df['PetalLengthCm'])


# In[46]:


plt.boxplot(df['PetalWidthCm'])


# In[6]:


import seaborn as sns

# Calculate the correlation matrix with numeric_only parameter
correlation_matrix = df.corr(numeric_only=True)

# Create a heatmap of the correlation matrix
sns.heatmap(correlation_matrix)


# In[19]:


sns.pairplot(df,hue='Species')


# # 3.Data Preparation Sepereating Input Columns And The Output Column

# In[92]:


df.drop('Id',axis=1,inplace=True)


# In[87]:


sp={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}


# In[90]:


df.Species=[sp[i] for i in df.Species]


# In[89]:


df


# In[69]:


X=df.iloc[:,0:4]


# In[70]:


X


# In[71]:


y=df.iloc[:,4]


# In[72]:


y


# # Split datset into train and test sets:

# In[95]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
print(y_test)


# In[26]:


np.unique(y_test, return_counts=True)


# In[84]:


# Check the columns in X_train and X_test
print("Columns in X_train:", X_train.columns)
print("Columns in X_test:", X_test.columns)

# Ensure that the columns in X_test match X_train
# You may need to preprocess X_test to align with X_train's features if needed.


# # Training Model

# # Model1 using Linear Regression

# In[79]:


from sklearn.linear_model import LinearRegression

# Create a Linear Regression model
model = LinearRegression()


# In[80]:


model.fit(X,y)


# In[81]:


model.coef_


# In[82]:


model.intercept_


# # Making Prediction

# In[96]:


y_pred1=model.predict(X_test)


# In[100]:


print("Mean squared error: %2f" %np.mean((y_pred1 - y_test)** 2))
print(accuracy_score(y_test,y_pred1)*100)


# # Model2:Support Vector Machine Algorithm
# 

# In[98]:


#Support Vector machine Algorithm
from sklearn.svm import SVC
#model fit
model_svc = SVC()
model_svc.fit(X_train, y_train)


# In[101]:


y_pred2=model_svc.predict(X_test)
print(accuracy_score(y_test,y_pred2)*100)


# # Model3 using Decision Tree Classifier

# In[103]:


from sklearn.tree import DecisionTreeClassifier 
model_DTC=DecisionTreeClassifier()
model_DTC.fit(X_train,y_train)


# In[104]:


y_pred3=model_svc.predict(X_test)
print(accuracy_score(y_test,y_pred3))


# In[108]:


from sklearn.metrics import classification_report

# Assuming you have the true labels in y_test and predicted labels in y_pred2
print(classification_report(y_test, y_pred3))


# In[ ]:




