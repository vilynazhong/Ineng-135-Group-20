#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA


# In[4]:


pip install seaborn


# In[5]:


df_precious_metal = pd.read_csv('londonfixes-current-clean_1990_.csv',index_col=0, parse_dates=True, usecols = ['Date','Gold AM Fix','Silver Fix','Platinum AM Fix','Palladium AM Fix'])
# why log? https://www.researchgate.net/post/What-is-the-best-way-to-scale-parameters-before-running-a-Principal-Component-Analysis-PCA
df_precious_metal = np.log(df_precious_metal)

year_list = df_precious_metal.index.year.unique().tolist()
n_components = 3
pca = PCA(n_components=n_components)
window_length = 10
col_PC = ['PC1','PC2','PC3']
col_PC = col_PC[:n_components]
splits = {'train': [], 'test': []}
df_explained_var = pd.DataFrame(data = [], columns = col_PC)
PC1_loadings = pd.DataFrame(data = [], columns = df_precious_metal.columns)
PC2_loadings = pd.DataFrame(data = [], columns = df_precious_metal.columns)
PC3_loadings = pd.DataFrame(data = [], columns = df_precious_metal.columns)
df_hat = pd.DataFrame(index = df_precious_metal.index, columns = df_precious_metal.columns)


# In[6]:


# PCA decomposition
for idx, yr in enumerate(year_list[:-window_length]):
    train_yr = year_list[idx:idx+window_length]
    test_yr = [year_list[idx+window_length]]
    print('TRAIN: ', train_yr, 'TEST: ',test_yr)
    
    splits['train'].append(df_precious_metal.loc[df_precious_metal.index.year.isin(train_yr), :])
    splits['test'].append(df_precious_metal.loc[df_precious_metal.index.year.isin(test_yr), :])
    
    traning_set = splits['train'][-1]
    pca.fit(traning_set)
    loadings = pd.DataFrame(pca.components_.T, columns=col_PC, index=df_precious_metal.columns)
    
    df_explained_var.loc[year_list[idx],:] = pca.explained_variance_ratio_
    PC1_loadings.loc[year_list[idx],:] = loadings.PC1.T
    PC2_loadings.loc[year_list[idx],:] = loadings.PC2.T
    PC3_loadings.loc[year_list[idx],:] = loadings.PC3.T
    
    testing_set = splits['test'][-1]
    df_transformed = pca.transform(testing_set)
    index = df_precious_metal.index[df_precious_metal.index.year.isin(test_yr)]
    df_hat.loc[index,:] = pd.DataFrame(data = pca.inverse_transform(df_transformed), columns = df_precious_metal.columns, index=index)
    


# In[7]:


df_explained_var


# In[8]:


PC1_loadings


# In[9]:


ax = df_precious_metal.loc[:,'Gold AM Fix'].plot()
df_hat.loc[:,'Gold AM Fix'].plot(ax=ax)
df_rolling_mean = df_precious_metal.rolling(window_length*252).mean()


# In[10]:


plt.plot(df_precious_metal.loc[:,'Gold AM Fix'] - df_hat.loc[:,'Gold AM Fix'])


# ### Binary prediction

# In[13]:


## backtest
df_hat = df_hat.dropna()
df_buy = df_precious_metal.loc[df_hat.index,'Gold AM Fix'] < df_hat.loc[df_hat.index,'Gold AM Fix'] - 0.1
hold_len = 10
df_up_after_hold_len = df_precious_metal.loc[df_hat.index,'Gold AM Fix'] < df_precious_metal.loc[df_hat.index,'Gold AM Fix'].shift(-hold_len)
print('Original up rate after {0} days: {1:d} buys, {2:.2f}%'.format(hold_len,df_up_after_hold_len.shape[0],sum(df_up_after_hold_len)/df_up_after_hold_len.shape[0]*100))
print('PCA predicted up rate after {0} days: {1:d} buys, {2:.2f}%'.format(hold_len,sum(df_buy),sum(df_buy & df_up_after_hold_len)/sum(df_buy)*100))
df_rolling = df_precious_metal.rolling(hold_len).mean()
df_MA_buy = df_precious_metal.loc[df_hat.index,'Gold AM Fix'] < df_rolling.loc[df_hat.index,'Gold AM Fix']
print('MA up rate after {0} days: {1:d} buys, {2:.2f}%'.format(hold_len,sum(df_MA_buy),sum(df_MA_buy & df_up_after_hold_len)/sum(df_MA_buy)*100))


# In[14]:


df_sell = df_precious_metal.loc[df_hat.index,'Gold AM Fix'] > df_hat.loc[df_hat.index,'Gold AM Fix'] + 0.17
hold_len = 10
df_down_after_hold_len = df_precious_metal.loc[df_hat.index,'Gold AM Fix'] > df_precious_metal.loc[df_hat.index,'Gold AM Fix'].shift(-hold_len)
print('Original down rate after {0} days: {1:.2f}%'.format(hold_len,sum(df_down_after_hold_len)/df_down_after_hold_len.shape[0]*100))
print('PCA predicted up rate after {0} days: {1:d} sells, {2:.2f}%'.format(hold_len,sum(df_sell),sum(df_sell & df_down_after_hold_len)/sum(df_sell)*100))


# more detailed analysis using confusion matrix
# ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

# In[ ]:




