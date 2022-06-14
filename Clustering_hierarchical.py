#!/usr/bin/env python
# coding: utf-8

# ### Import the libraries
# 

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import scipy.cluster.hierarchy as sch ##creating dendogram
from sklearn.cluster import AgglomerativeClustering  ##creating cluster


# In[6]:


### load the data
data=pd.read_csv("crime_data.csv")
data.head()


# In[ ]:


## Find correlation
data.corr()


# In[ ]:


data.info()


# In[5]:


data[data.duplicated()].shape


# In[6]:


import seaborn as sns
sns.pairplot(data)


# # Check for null values
# 
# 

# In[10]:


data.isnull().sum()


# ### Normalise the data to avoid scaling effect
# 

# In[3]:


#Normalisation function 

def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[7]:


##Normalise the data only for the numerical part and leave the country name

df_norm = norm_fun(data.iloc[:,1:])


# ### Create dendogram using single linkage method

# In[9]:


dendrogram= sch.dendrogram(sch.linkage(df_norm, method='single'))


# ### Create clusters

# In[10]:


hc= AgglomerativeClustering(n_clusters=4, affinity= 'euclidean', linkage='single')


# ### Save clusters for chart`

# In[12]:


y_hc= hc.fit_predict(df_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[15]:


data['h_clusterid']=Clusters


# In[16]:


data


# In[ ]:




