#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("monthly_csv.csv")
df.head(10)


# In[3]:


df


# In[4]:


# Setup the data
df.Date = pd.to_datetime(df.Date)
X = df.Date.values.reshape(-1, 1)
y = df.Price.values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state=100)

model = linear_model.LinearRegression()

print ('Number of samples in training data:',len(x_train))
print ('Number of samples in validation data:',len(x_test))

# fitting the data
model.fit(x_train, y_train)
#model.fit(X, y)
print ('INTERCEPT:',model.intercept_)
print ('COEFFICIENTS:\n',model.coef_)

#calculate the mean square error
training_error=np.mean((model.predict(x_train.astype(float)) - y_train) ** 2)
print("Training Error (Mean squared error) :",training_error)

testing_error=np.mean((model.predict(x_test.astype(float)) - y_test) ** 2)
print("Testing Error (Mean squared error) :",testing_error)

# make a prediction along the time of all the data
y_pred = model.predict(X.astype(float))

plt.scatter(X,y)
plt.xlabel('Date')
plt.ylabel('Price $')
plt.plot(X,y_pred,'-r')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




