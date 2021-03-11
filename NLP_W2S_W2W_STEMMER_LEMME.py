#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
# nltk.download()
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords


# In[2]:


Pharagraph = 'Paragraphs are the building blocks of papers. Many students define paragraphs in terms of length: a paragraph is a group of at least five sentences, a paragraph is half a page long, etc. In reality, though, the unity and coherence of ideas among sentences is what constitutes a paragraph. A paragraph is defined as “a group of sentences or a single sentence that forms a unit” (Lunsford and Connors 116). Length and appearance do not determine whether a section in a paper is a paragraph. For instance, in some styles of writing, particularly journalistic styles, a paragraph can be just one sentence long. Ultimately, a paragraph is a sentence or group of sentences that support one main idea. In this handout, we will refer to this as the “controlling idea,” because it controls what happens in the rest of the paragraph'


# In[3]:


sentents = nltk.sent_tokenize(Pharagraph)


# In[4]:


sentents


# In[5]:


word = nltk.word_tokenize(Pharagraph)
word


# In[6]:


stemmer = PorterStemmer()
for i in range(len(sentents)):
    words = nltk.word_tokenize(sentents[i])
    words = [stemmer.stem(word) for word in words if word  not in set(stopwords.words('english'))]
    sentents[i]= ' '.join(words)


# In[ ]:




