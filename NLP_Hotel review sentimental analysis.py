#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_score,recall_score


# In[6]:


data = pd.read_csv(r"C:\Users\dell\Downloads\train.csv (1)\train.csv")


# In[7]:


data.head()


# In[8]:


data.shape


# In[9]:


data.info()


# In[11]:


data.describe().transpose()


# In[16]:


count = data.isnull().sum().sort_values(ascending= False)
percentage = ((data.isnull().sum()/len(data)*100)).sort_values(ascending= False)
missing_data = pd.concat([count,percentage],axis=1,keys = ['count','Percentage'])
print('count and percentage of missing values for the column: ')
missing_data


# In[20]:


print('percentage for defualt\n')
print(round(data.Is_Response.value_counts(normalize=True)*100,2))
round(data.Is_Response.value_counts(normalize=True)*100,2).plot(kind='bar')
plt.title('Percentage Distributions by review type')
plt.show()


# In[21]:


data.drop(columns=['User_ID','Browser_Used','Device_Used'],inplace=True)


# In[26]:


def text_clean(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w','',text)
    return text
cleaned1 = lambda x: text_clean(x)
    


# In[29]:


data["cleaned_description"]= pd.DataFrame(data.Description.apply(cleaned1))
data.head()


# In[31]:


def text_clean1(text):
    text = re.sub('[''""...]','',text)
    text = re.sub('\n','',text)
    return text
cleaned2 = lambda x: text_clean1(x)


# In[36]:


data['cleaned_Description_new'] = pd.DataFrame(data['cleaned_description'].apply(cleaned2))
data.head()


# In[38]:


x =  data.cleaned_Description_new
y =  data.Is_Response


# In[39]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.10,random_state = 42)


# In[40]:


print('x_train :',len(x_train))
print('x_test :',len(x_test))
print('y_train :',len(y_train))
print('y_test :',len(y_test))


# In[42]:


tvec = TfidfVectorizer()
clf = LogisticRegression(solver='lbfgs')


# In[46]:


model = Pipeline([('vectorizer',tvec),('classifier',clf)])
model.fit(x_train,y_train)


# In[50]:


prediction = model.predict(x_test)
confusion_matrix(prediction,y_test)


# In[51]:


accuracy_score(y_test,prediction)


# In[54]:


recall_score(y_test,prediction,average='weighted')


# In[55]:


precision_score(y_test,prediction,average='weighted')


# In[58]:


example = ["i'm satisfied"]
result = model.predict(example)
result


# In[59]:


example = ["i' unhappy"]
result = model.predict(example)
result


# In[60]:


example = ["i' frustrated"]
result = model.predict(example)
result


# In[ ]:




