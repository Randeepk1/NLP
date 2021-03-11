#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score


# In[2]:


msg =  pd.read_csv(r"C:\Users\dell\Downloads\smsspamcollection\SMSSpamCollection",sep='\t',names=['label','Message'])


# In[3]:


msg


# In[4]:


ps = PorterStemmer()
corpus = []
for i in range(len(msg)):
    review = re.sub('[^A-Z,a-z]',' ', msg['Message'][i])
    review  = review.lower()
    review = review.split()
    review = [ps.stem(word)for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
#     print(review)
    corpus.append(review)
cv = CountVectorizer(max_features=2500)
x =cv.fit_transform(corpus).toarray()
x


# In[7]:


y = pd.get_dummies(msg['label'])
y = y.iloc[:,1].values


# In[11]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state =42)


# In[13]:


spam_detect_model = MultinomialNB().fit(x_train,y_train)


# In[15]:


y_pred = spam_detect_model.predict(x_test)
y_pred


# In[17]:


confusion_matrix(y_test,y_pred)


# In[19]:


accuracy_score(y_test,y_pred)

