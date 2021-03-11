#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from gensim.models import Word2Vec
import re
from nltk.corpus import stopwords


# In[2]:


# !pip install -U gensim


# In[3]:


Pharagraph = 'Paragraphs are the building blocks of papers. Many students define paragraphs in terms of length: a paragraph is a group of at least five sentences, a paragraph is half a page long, etc. In reality, though, the unity and coherence of ideas among sentences is what constitutes a paragraph. A paragraph is defined as “a group of sentences or a single sentence that forms a unit” (Lunsford and Connors 116). Length and appearance do not determine whether a section in a paper is a paragraph. For instance, in some styles of writing, particularly journalistic styles, a paragraph can be just one sentence long. Ultimately, a paragraph is a sentence or group of sentences that support one main idea. In this handout, we will refer to this as the “controlling idea,” because it controls what happens in the rest of the paragraph'


# In[4]:


text = re.sub(r'\[[0-9]*\]',' ',Pharagraph)
text = re.sub('[^A-Z,a-z]',' ', Pharagraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d ',' ',text)
text = re.sub(r'\s+',' ',text)
text


# In[14]:


sentences = nltk.sent_tokenize(text)
sentences


# In[16]:


words = [nltk.word_tokenize(sentences) for sentences in sentences ]
words


# In[18]:


for i in range(len(words)):
    word[i] = [word for word in word[i] if word not in stopwords.words('english')]


# In[26]:


model = Word2Vec(words,min_count=1)
word = model.wv.vocab
vector = model.wv['long']
vector.shape
similar = model.wv.most_similar('long')
similar


# In[ ]:




