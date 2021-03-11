#!/usr/bin/env python
# coding: utf-8

# In[4]:


import nltk
from gensim.models import Word2Vec
import re
from nltk.corpus import stopwords


# In[5]:


Pharagraph = 'Paragraphs are the building blocks of papers. Many students define paragraphs in terms of length: a paragraph is a group of at least five sentences, a paragraph is half a page long, etc. In reality, though, the unity and coherence of ideas among sentences is what constitutes a paragraph. A paragraph is defined as “a group of sentences or a single sentence that forms a unit” (Lunsford and Connors 116). Length and appearance do not determine whether a section in a paper is a paragraph. For instance, in some styles of writing, particularly journalistic styles, a paragraph can be just one sentence long. Ultimately, a paragraph is a sentence or group of sentences that support one main idea. In this handout, we will refer to this as the “controlling idea,” because it controls what happens in the rest of the paragraph'


# In[6]:


text = re.sub(r'\[[0-9]*\]',' ',Pharagraph)
text = re.sub('[^A-Z,a-z]',' ', Pharagraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d ',' ',text)
text = re.sub(r'\s+',' ',text)
text


# In[7]:


sentences = nltk.sent_tokenize(text)
sentences


# In[8]:


words = [nltk.word_tokenize(sentences) for sentences in sentences ]
words


# In[9]:


my_model =  Word2Vec(words,min_count=1)


# In[10]:


my_model


# In[11]:


words = list(my_model.wv.vocab)


# In[12]:


words


# In[15]:


my_model['one']


# In[19]:


my_model.most_similar('one')


# In[ ]:




