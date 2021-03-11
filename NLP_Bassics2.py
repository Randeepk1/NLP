#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.corpus import brown
from nltk.book import *
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk import pos_tag,ne_chunk


# In[2]:


brown.categories()


# In[3]:


brown.words()


# In[4]:


data = 'A paragraph is a self-contained unit of discourse in writing dealing with a particular point or idea. A paragraph consists of one or more sentences. Though not required by the syntax of any language, paragraphs are usually an expected part of formal writing, used to organize longer prose'


# In[5]:


stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(data)
word_tokens


# In[6]:


filterd_sent = [w for w in word_tokens if not w in stop_words]
filterd_sent = []
for w in word_tokens:
    if w not in stop_words:
        filterd_sent.append(w)
print(filterd_sent)


# In[7]:


ps = PorterStemmer()
words = word_tokenize(data)
for w in words:
    print(ps.stem(w))


# In[8]:


lem = WordNetLemmatizer()


# In[9]:


stop_words = set(stopwords.words('english'))
tokenized = sent_tokenize(data)
for i in tokenized:
    word_list = word_tokenize(i)
    word_list = [w for w in word_list if not w in stop_words]
    tagged =  pos_tag(word_list)
    print(tagged)


# In[10]:


token = word_tokenize(data)
taged_sent = pos_tag(token)
ne_chunked_sent =  ne_chunk(taged_sent)
ne_chunked_sent


# In[ ]:





# In[ ]:




