#!/usr/bin/env python
# coding: utf-8

# In[1]:


length = 5
breadth = 2
area = length * breadth
print('Площадь равна', area)
print('Периметр равен', 2 * (length + breadth))


# In[89]:


import pandas as pd
train_set = pd.read_csv("train.csv")
train_set.head(10)


# In[90]:


train_set.ADDRESS.value_counts()


# In[71]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20 ,5))
train_set.ADDRESS = train_set.ADDRESS.apply(lambda x: x.split(',')[-1])
train_set.ADDRESS.value_counts()[:20].plot(kind='bar')
plt.show()


# In[72]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20 ,20))
train_set.groupby(['ADDRESS']).sum().SQUARE_FT.nlargest(10).plot(kind='pie')
plt.show()


# In[79]:


plt.figure(figsize=(20 ,8))
train_set.groupby(['ADDRESS']).sum().UNDER_CONSTRUCTION.nlargest(10).plot(kind='bar')
plt.show()


# In[91]:


top_cities = train_set.ADDRESS.value_counts().nlargest(15).index
train_set.ADDRESS =train_set.ADDRESS.apply(lambda x: x if x in top_cities else 'Other')
train_set.ADDRESS.value_counts()


# In[67]:


import pandas as pd
train_set = pd.read_csv("train.csv")
train_set.info()


# In[16]:


train_set.shape


# In[78]:


import numpy as np
train_set.TARGET(PRICE_IN_LACS)=train_set.TARGET(PRICE_IN_LACS).astype(np.int64)
train_set.head(10)


# In[77]:


from sklearn import preprocessing

def label_encoder(train_set, column_name):
    label_encoder = preprocessing.LabelEncoder()

    train_set[column_name]= label_encoder.fit_transform(train_set[column_name])
    print(column_name)
    for i in range(len(train_set[column_name].unique())):
        print("For {} : {}".format(i, label_encoder.inverse_transform([i])))
    print('-'*10)
    print(train_set[column_name].value_counts())
    print('-'*10)
    
    return train_set[column_name], label_encoder
train_set['BHK_OR_RK'], label_encoder_posted_by = label_encoder(train_set, 'BHK_OR_RK')
train_set


# In[99]:


def label_encoder(df, column_name):
    label_encoder = preprocessing.LabelEncoder()

    df[column_name]= label_encoder.fit_transform(df[column_name])
    print(column_name)
    for i in range(len(df[column_name].unique())):
        print("For {} : {}".format(i, label_encoder.inverse_transform([i])))
    print('-'*10)
    print(df[column_name].value_counts())
    print('-'*10)
    
    return df[column_name], label_encoder


train_set['POSTED_BY'], label_encoder_posted_by = label_encoder(train_set, 'POSTED_BY')
train_set


# In[93]:


train_set['BHK_NO.'].value_counts()


# In[ ]:





# In[98]:



train_set['ADDRESS'], label_encoder_posted_by = label_encoder(train_set, 'ADDRESS')


train_set


# In[ ]:





# In[ ]:





# In[ ]:




