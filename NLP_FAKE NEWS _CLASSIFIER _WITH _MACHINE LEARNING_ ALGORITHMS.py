#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import itertools
from sklearn.linear_model import PassiveAggressiveClassifier


# In[2]:


data = pd.read_csv(r"G:\DATASETS\New folder\train.csv")


# In[3]:


data.head()


# In[4]:


x = data.drop('label',axis=1)


# In[5]:


x.head(7)


# In[6]:


y = data['label']
y.head(5)


# In[7]:


data.shape


# In[8]:


data.isnull().sum().sum()


# In[9]:


data = data.dropna()
data.isnull().sum().sum()


# In[10]:


data.reset_index(inplace=True)


# In[11]:


data.shape


# In[12]:


data['title'][6]


# In[13]:


ps = PorterStemmer()
corpus = []
for i in range(len(data)):
    review = re.sub('[^A-Z,a-z]',' ', data['title'][i])
    review  = review.lower()
    review = review.split()
    review = [ps.stem(word)for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# In[14]:


cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
x = cv.fit_transform(corpus).toarray()


# In[15]:


corpus[3]


# In[16]:


x.shape


# In[17]:


y=data['label']


# In[18]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state= 0)


# In[19]:


x_train.shape


# In[20]:


cv.get_feature_names()[:20]


# In[21]:


corpus


# In[22]:


cv.get_params()


# In[23]:


count_data = pd.DataFrame(x_train,columns=cv.get_feature_names())
count_data.head()


# In[42]:


def plot_confusion_matrix(cm,classes,normalize = False,title = 'confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axix=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("confusion matrix")
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]),range((cm.shape[1]))):
        plt.text(j,i,cm[i,j],
                horizontalalignment='center',
#                 color = "white" if cm [i,j] > thresh else "black"
                color = "white" if cm[i,j]> thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[43]:


clf = MultinomialNB()


# In[44]:


clf.fit(x_train,y_train)
pred = clf.predict(x_test)
score = metrics.accuracy_score(y_test,pred)
print("accuracy: %0.3f"% score)
cm= metrics.confusion_matrix(y_test,pred)
plot_confusion_matrix (cm,classes = ['FAKE','REAL'])


# In[45]:


clf.fit(x_train,y_train)
pred = clf.predict(x_test)
score = metrics.accuracy_score(y_test,pred)
score


# In[46]:


y_train.shape


# In[55]:


model = PassiveAggressiveClassifier(n_iter_no_change = 50)
model.fit(x_train,y_train)
pred= model.predict(x_test)
score = accuracy_score(y_test,pred)
score


# In[56]:


cm= metrics.confusion_matrix(y_test,pred)
plot_confusion_matrix (cm,classes = ['FAKE','REAL'])


# In[59]:


previous_score = 0
clf1 = MultinomialNB(alpha=0.1)
for alpha in np.arange(0,1,0.1):
    sub_clsifr = MultinomialNB(alpha=alpha)
    sub_clsifr.fit(x_train,y_train)
    y_pred = sub_clsifr.predict(x_test)
    score = accuracy_score(y_test,y_pred)
    if score>previous_score:
        clf2 = sub_clsifr
    print("Alpha:{},score:".format(alpha,score))


# In[61]:


feture_name = cv.get_feature_names()
feture_name


# In[62]:


clf2.coef_


# In[63]:


sorted(zip(clf2.coef_[0],feture_name),reverse=True)[:20]


# In[64]:


sorted(zip(clf2.coef_[0],feture_name))[:3400]


# In[ ]:




