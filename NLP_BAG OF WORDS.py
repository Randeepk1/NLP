#!/usr/bin/env python
# coding: utf-8

# In[20]:


import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


# In[15]:


Pharagraph = 'Paragraphs are the building blocks of papers. Many students define paragraphs in terms of length: a paragraph is a group of at least five sentences, a paragraph is half a page long, etc. In reality, though, the unity and coherence of ideas among sentences is what constitutes a paragraph. A paragraph is defined as “a group of sentences or a single sentence that forms a unit” (Lunsford and Connors 116). Length and appearance do not determine whether a section in a paper is a paragraph. For instance, in some styles of writing, particularly journalistic styles, a paragraph can be just one sentence long. Ultimately, a paragraph is a sentence or group of sentences that support one main idea. In this handout, we will refer to this as the “controlling idea,” because it controls what happens in the rest of the paragraph'


# In[18]:


ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentence  = nltk.sent_tokenize(Pharagraph)
corpus =  []
for i in range(len(sentence)):
    review = re.sub('[^A-Z,a-z,]',' ',sentence[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word)for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# In[25]:


cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()


# In[26]:


x


# In[ ]:




