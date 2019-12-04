
# coding: utf-8

# # Load and Visualize Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


# import dataset
df = pd.read_csv('data/bank-additional-full.csv',sep=';',)


# In[501]:


df.info() # no missing values for all the features


# In[4]:


df.head()


# In[502]:


df.describe() # basic descriptive statistics


# In[3]:


df['y'] = df['y'].map({'no':0, 'yes':1}) # binary encoding of class label


# In[7]:


df.hist(figsize=(12,10)) # display numerical feature distribution


# In[8]:


df['y'].value_counts() # dataset is imbalanced with majority of class label as "no".


# In[9]:


# dataset is imbalanced with majority of class label as "no".
df['y'].value_counts()/len(df)


# In[508]:


df.corr() # correlation matrix analysis to show r2 value


df2 = df[["age", "duration", "campaign" ]]

# In[511]:


# scatter matrix visualization
from pandas.plotting import scatter_matrix
scatter_matrix(df2, figsize=(18,12))


# In[13]:


# visualize categorical features
categorical = ['education','contact','month']
for i in categorical:
    df[i].value_counts().plot(kind='bar',figsize = (10, 2),title=i)
    plt.show()


    