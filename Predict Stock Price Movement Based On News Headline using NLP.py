#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[2]:


data = pd.read_csv(r'G:\DATASETS\Stock_Dataa.csv',encoding='ISO-8859-1')
data


# In[3]:


train = data[data['Date']<'20150101']
test = data[data['Date']> '20141231']


# In[6]:


data = train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex =True,inplace =True)  #removing all unwanted characters


# In[7]:


list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
data.columns = new_index
data.head()


# In[8]:


for index in new_index:
    data[index]= data[index].str.lower()
data.head()


# In[10]:


headlines = []
for row in range (0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))  # all each every index and read and combine them into one paragraph 


# In[11]:


headlines[0]


# In[14]:


' '.join(str(x) for x in data.iloc[1,0:25])  # first index convert into a paragrarah for showing


# In[18]:


countvector = CountVectorizer(ngram_range=(2,2))               # bag of words
traindataset = countvector.fit_transform(headlines)


# In[19]:


rfc = RandomForestClassifier(n_estimators=200,criterion='entropy')
rfc.fit(traindataset,train['Label'])


# In[23]:


test_transform = []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
prediction = rfc.predict(test_dataset)


# In[27]:


matrix = confusion_matrix(test['Label'],prediction)
matrix
score = accuracy_score(test['Label'],prediction)
score
report = classification_report(test['Label'],prediction)
report

