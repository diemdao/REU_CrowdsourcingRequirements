#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# read the data in and print it out
df = pd.read_csv('requirements.csv')
df


# In[2]:


# see what is in the 'feature' column (mainly focus on the 'feature' column bc it's the ideas that the users put down)
y = df['feature']
y


# In[3]:


# tokenize, breaking down to each word in 'feature' column
from nltk.tokenize import word_tokenize

tokenized_word = df['feature'].apply(word_tokenize)
tokenized_word


# In[4]:


# create another column called tokenized, to see keep track of the process
df['Tokenized'] = tokenized_word

# change to lower characters
map(lambda x:x.lower(), df['Tokenized'])
df


# In[5]:


# gathering stopwords
import nltk
from nltk.corpus import stopwords

stop_words = nltk.corpus.stopwords.words('english')

newStopWords = ['smart', 'home', 'house', 'automate', 'I', 'A', 'My', 'smarter', 'system', 'Im', 'automatically', 'automatic', 'someone', 'anyone', 'person', 'whenever', 'wherever', 'specific']
stop_words.extend(newStopWords)

import string
stop_words.extend(string.punctuation)

print(stop_words)


# In[6]:


# filter out stopwords

filtered_sent = []
for sentence in tokenized_word:
    templist = []
    for each_word in sentence:
        if each_word not in stop_words:
            templist.append(each_word)
    filtered_sent.append(templist)

print("Tokenized Sentence:",tokenized_word)
print("Filtered Sentence:",filtered_sent)


# In[7]:


df['Filtered'] = filtered_sent
# make it to lower characters
map(lambda x:x.lower(), df['Filtered'])
df['Filtered']


# In[8]:


# Stemming
# reducing inflection in words to their root forms
from nltk.stem import PorterStemmer

# remove morphological affixes from words
ps = PorterStemmer()

stemmed_words = []
for sentence in filtered_sent:
    templist = []
    for each_word in sentence:
        templist.append(ps.stem(each_word))
    stemmed_words.append(templist)

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)


# In[9]:


# put stemmed_words as a new column called Stemmed
df['Stemmed'] = stemmed_words
df


# In[10]:


# POS (Part-of-Speed) tagging (NOUN, PRONOUN, ADJECTIVE, VERB, ADVERBS, etc.)
POS = []
for sentence in stemmed_words:
    POS.append(nltk.pos_tag(sentence))
    
print(POS)


# In[11]:


df['POS'] = POS
df


# In[12]:


# Make the list of lists words in Stemmed turned to a list of string so we can do TD-IDFvectors
df['StringTest'] = df['Stemmed'].apply(', '.join)

Teststring = df['StringTest']


# In[13]:


# remove numbers after stemming and changed to a list of string

import re    # import regular expression, to remove numbers in each requirements
  
def remove(list): 
    pattern = '[0-9]'
    list = [re.sub(pattern, '', i) for i in list] 
    return list
  
# Driver code  
print(remove(Teststring)) 

removeN = remove(Teststring)


# In[14]:


df['cleaned_feature'] = removeN
df


# In[15]:


# use TF-IDF vetorizer to count important words in the data frame
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['cleaned_feature'])
pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names())


# In[16]:


tfidf_vectorizer = TfidfVectorizer()
# put TD-IDFvectors as a new column
df['TD-IDFvectors'] = list(tfidf_vectorizer.fit_transform(df['cleaned_feature']).toarray())
df


# In[17]:


# this is the same thing as box 15, but store it as dataF for later use
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_feature'])
dataF = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
dataF


# In[18]:


dataF.describe()  # descriptive statistics of the words


# In[19]:


# try to Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(dataF, dataF))


# In[20]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = dataF

sse = {}

# run 100 k-means and calculate Sum of Squared Errors (SSE)
for k in range(1, 100):
#   kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
    kmeans = KMeans(n_clusters=k).fit(data)
    data["clusters"] = kmeans.labels_
    # print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center


# In[21]:


# Elbow method, you can see the SSE and the Number of cluster K
# Elbow method shows the optimal number of K clusters

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster K")
plt.ylabel("SSE")
plt.show()


# In[22]:


# from sklearn.cluster import KMeans

# after finding the optimal point of clusters (from the graph above), it's around 55 clusters

# This shows the 55 most common ideas and concepts from all the requirements
number_of_clusters = 55
km = KMeans(n_clusters=number_of_clusters)
km.fit(X)

print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(number_of_clusters):
    top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
    print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))


# In[23]:


km = KMeans(n_clusters=55).fit(data)

cluster_map = pd.DataFrame()
cluster_map['cluster'] = km.labels_
cluster_map['data_index'] = data.index.values
cluster_map['text'] = df['cleaned_feature']

# display the groups of subsets, later use for offline phase (initial training phase)

# display what requirements are in cluster 1
cluster_map[cluster_map.cluster == 1]


# In[24]:

# --------------------
# I create each cluster once at a time and same it in a file, bc it's better to keep track this way. 
# Another way is you can make this progess faster saving 55 files (55 clusters) by using for loop and increment it
# --------------------

# cluster0 = cluster_map[cluster_map.cluster == 0]
# cluster0.to_csv('clusters_Diem/cluster_0.csv')


# In[25]:


# cluster1 = cluster_map[cluster_map.cluster == 1]
# cluster1.to_csv('clusterOne.csv')


# In[26]:


# display what requirements are in cluster 2
cluster_map[cluster_map.cluster == 2]


# In[27]:


# cluster2 = cluster_map[cluster_map.cluster == 2]
# cluster2.to_csv('clusterTwo.csv')


# In[28]:


# display what requirements are in cluster 3
cluster_map[cluster_map.cluster == 3]


# In[29]:


# cluster3 = cluster_map[cluster_map.cluster == 3]
# cluster3.to_csv('clusterThree.csv')


# In[30]:


cluster_map[cluster_map.cluster == 4]


# In[31]:


# cluster4 = cluster_map[cluster_map.cluster == 4]
# cluster4.to_csv('clusterFour.csv')


# In[32]:


cluster_map[cluster_map.cluster == 5]


# In[33]:


# cluster5 = cluster_map[cluster_map.cluster == 5]
# cluster5.to_csv('clusterFive.csv')


# ---------------- Countinue to create the process above to save 55 files (55 clusters).-------------

# In[83]:

# ---------Finally----------
# try to merge 2 files to get the different phases of questions

df2 = pd.read_csv('users.csv')
df2


# In[84]:


# the user_id in df from requirements.csv is assigned as id,
# so when inner join id from users.csv it's easier because each user_id have a phase and can have more than one requirements
df['id'] = df['user_id']


# In[85]:


innerTest = pd.merge(df, df2, on = 'id')

ID_Phase = innerTest[['id', 'created_phase', 'cleaned_feature']]
ID_Phase


# In[86]:


ID_Phase.to_csv('ID_Phase.csv')


# In[ ]:




