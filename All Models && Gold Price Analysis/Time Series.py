#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df1=pd.read_csv('annual_csv.csv')
df2=pd.read_csv('monthly_csv.csv')


# In[3]:


print(df1.head())
print(df2.head())


# In[4]:


#no obvious seasonal pattern is observed
df1.plot(figsize=(16,5),grid=True)


# In[5]:


df2.plot(figsize=(16,5),grid=True)


# In[6]:


#Very high p-value. The data is not stationary
from statsmodels.tsa.stattools import adfuller
adfuller(df1['Price'])


# In[7]:


#very high p-value. The data is not stationary
adfuller(df2['Price'])


# In[8]:


#How can we make these data stationary
df1_stationary =df1['Price'].diff()
df2_stationary =df1['Price'].diff()


# In[9]:


df1_stationary.plot()


# In[10]:


df2_stationary.plot()


# In[11]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[12]:


plot_acf(df1['Price'])
plot_pacf(df1['Price'])


# In[13]:


plot_acf(df2['Price'])
plot_pacf(df2['Price'])


# In[45]:


#Fitting models

Train1=df1.iloc[:35,1]
Test1=df1.iloc[35:,1]
Train2=df2.iloc[:424,1]
Test2=df2.iloc[424:,1]


# In[46]:


Train1=[x for x in Train1]
Test1=[x for x in Test1]
Train2=[x for x in Train2]
Test2=[x for x in Test2]


# In[47]:


from statsmodels.tsa.arima_model import ARIMA


# In[19]:


pred=[]
low_bound=[]
up_bound=[]

for data in range(len(Test1)):
    model=ARIMA(Train1,order=(1,1,0))
    model_fit=model.fit(disp=0)
    out=model_fit.forecast()
    
    result=out[0]
    lower=out[2][0][0]
    upper=out[2][0][1]
    
    pred.append(result)
    low_bound.append(lower)
    up_bound.append(upper)
    
    obs = Test1[data]
    Train1.append(obs)
    
    print('Actual Value:',obs,'Prediction',result)

#plot

plt.plot(Test1,color='black',linewidth=1)
plt.plot(low_bound,color='red',linewidth=1)
plt.plot(up_bound,color='green',linewidth=1)
plt.plot(pred,linewidth=1)
plt.show()


# In[86]:


pred=[]
low_bound=[]
up_bound=[]

for data in range(len(Test2)):
    model=ARIMA(Train2,order=(0,2,1))
    model_fit=model.fit(disp=0)
    out=model_fit.forecast()
    
    result=out[0]
    lower=out[2][0][0]
    upper=out[2][0][1]
    
    pred.append(result)
    low_bound.append(lower)
    up_bound.append(upper)
    
    obs = Test2[data]
    Train2.append(obs)
    
    print('Actual Value:',obs,'Prediction',result)

#plot

plt.plot(Test2,color='black',linewidth=1)
plt.plot(low_bound,color='red',linewidth=1)
plt.plot(up_bound,color='green',linewidth=1)
plt.plot(pred,linewidth=1)
plt.show()


# In[6]:


#try automation

get_ipython().system('pip install pmdarima')


# In[7]:


import pmdarima as pm


# In[62]:


df22=df2
df22.head()
#df22['Date']=pd.to_datetime(df22['Date'])
#df22.set_index('Date',inplace=True)
#df22.head()
Train22=df22[1:424]
Test22=df22[424:len(df22)]

model=pm.auto_arima(Train22['Price'],m=12,seasonal=True,start_q=0,star_p=0,max_order=5,error_action='ignore',stepwise=True,trace=True)


# In[63]:


model.summary()


# In[64]:


model.plot_diagnostics()


# In[76]:


prediction=pd.DataFrame(model.predict(n_periods=423),index=Test22.index)
prediction.columns=['predicted_sales']


# In[77]:


prediction


# In[78]:


plt.plot(prediction)
plt.plot(Train22)
plt.plot(Test22)


# In[48]:


prediction=pd.DataFrame(model.predict(n_periods=423),index=Test22.index)
prediction.columns=['predicted_sales']


# In[84]:


model=pm.auto_arima(df22['Price'],m=12,seasonal=False,start_q=0,star_p=0,max_order=5,error_action='ignore',stepwise=True,trace=True)


# In[85]:


prediction=pd.DataFrame(model.predict(n_periods=423),index=Test22.index)
prediction.columns=['predicted_sales']

plt.plot(prediction)
plt.plot(Train22)
plt.plot(Test22)


# In[82]:


prediction


# In[ ]:




