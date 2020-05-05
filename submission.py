"""
Bengali Newspaper data available on Kaggle. URL: https://www.kaggle.com/furcifer/bangla-newspaper-dataset 
Bengali Stopword is provided here. 
Replace the DATA_PATH, STOPWORD_PATH with data and stopword location. 
"""
DATA_PATH = "./data/data.json"
STOPWORD_PATH = "./data/stopwords-bn.txt"

# Misc
import json
import re
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import pprint
import pickle
import editdistance
from collections import Counter
# NLTK
import nltk
# Gensim
import gensim
from gensim.models import Word2Vec
# Scikit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,MultiLabelBinarizer
from sklearn.metrics import accuracy_score,hamming_loss
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation,Dense,Dropout,Embedding,Flatten,Conv1D,MaxPooling1D,LSTM,SimpleRNN,GlobalMaxPool1D
from keras import utils,layers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
# %tensorflow_version 1.x

# LOAD THE DATA
with open(DATA_PATH,'r') as jsonfile:
    data = jsonfile.read()

data = json.loads(data)

# REMOVE EXTRA COLUMNS
for i in data:
    if 'author' in i:
        del i['author']
    if 'published_date' in i:
        del i['published_date']
    if 'modification_date' in i:
        del i['modification_date']
    if 'comment_count' in i:
        del i['comment_count']
    if 'url' in i:
        del i['url']

df = pd.DataFrame(data[:],columns=data[0].keys())

# CLEAN UP FUNCTIONS
with open(STOPWORD_PATH,'r') as txtfile:
    lines = txtfile.readlines()
stopword_bn = [i.strip() for i in lines]
pattern = re.compile(u'[(.$))]|[,।‘’-]|\s\s+')

def rmv_punc(text):
    """ Returns input string without daris and commas
    Input:
        text: string.
    Output:
        string.
    """
    if isinstance(text,str):
        return pattern.sub(' ',text).strip()
def rmv_stopword(text):
    """Returns without stopwords"""
    if isinstance(text,str):
        return ' '.join([w for w in text.split() if w not in stopword_bn])

df['content'] = df['content'].apply(rmv_punc)
df['content'] = df['content'].apply(rmv_stopword)

# PREDEFINED TAGS SELECTION
N = 100
tags = list() # ALL TAGS
for i in range(len(df)):
    if isinstance(df.iloc[i,2],list):
        for t in df.iloc[i,2]:
            tags.append(t)
freq_tags = nltk.FreqDist(tags)
top_n_tags = [t[0] for t in freq_tags.most_common(N)] # MOST FREQUENT N TAGS

def remove_tags(tags):
    """
    REMOVE ANY ARTICLE THAT DOES NOT CONTAIN TOP N TAGS
    INPUT:
    tags. list of tags in a document.
    RETURNS:
    NONE. replaces the tags column for that document with None. 
    """
    if isinstance(tags,list):
        ts = []
        for t in tags:
            if t in top_n_tags:
                ts.append(t)
    else:
        return None
    return ts if len(ts)>0 else None


df.tag = df.tag.apply(lambda x: remove_tags(x)) 
df = df.dropna()

# PREPROCESSING
mlb = MultiLabelBinarizer()
mlb.fit(df.tag)
labels = mlb.classes_
maxlen = 200
nb_words = 50000
tk = Tokenizer(num_words=nb_words)
tk.fit_on_texts(df.content)
x = pad_sequences(tk.texts_to_sequences(df.content),maxlen=200)
y = mlb.transform(df.tag)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)

num_classes = N
class_weights = dict(freq_tags.most_common(N))
idx = 0
key = list(class_weights.keys())
for k in key:
    class_weights[k] = len(df) / class_weights[k]
    class_weights[idx] = class_weights.pop(k)
    idx += 1 

print("------------------------Vectorization Done--------------------")

# MODEL 
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

model = Sequential()
model.add(Embedding(nb_words, 100, input_length=maxlen))
model.add(LSTM(128))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_crossentropy'])
callbacks = [
    ReduceLROnPlateau(),
    EarlyStopping(patience=4)
]

from keras.utils import plot_model
history = model.fit(X_train, y_train,
                    class_weight=class_weights,
                    epochs=10,
                    batch_size=1000,
                    validation_data=(X_test,y_test),
                    callbacks=callbacks)

print("------------------------Training Complete--------------------")
def rmv_punc(text):
    if isinstance(text,str):
        return pattern.sub(' ',text).strip()
def rmv_stopword(text):
    if isinstance(text,str):
        return ' '.join([w for w in text.split() if w not in stopword_bn])
def stem(text):
    global run
    if isinstance(text,str):
        if (run%10000)==0:
            print("Running %d th article" %(run))
        run+=1
        return ' '.join([stemmer.stem_word(w) for w in text.split()])
def same_word(w1,w2):
    """
    Measures word similarity based on EditDistance Algorithm on the last indices of words.
    """
    dist = editdistance.eval(w1,w2)
    if len(w1)>2 and len(w2)>2 and dist<=6: # 6 is the length of গুলোতে, longest bibhokti
        
        t1 = w1[0:int(len(w1)/2)+1] # cutting in half
        t2 = w2[0:int(len(w1)/2)+1]
        dist2 = editdistance.eval(t1,t2)
        if dist2==0: # matching if first half of the words are same
            return True
    return False

def match_w_tag_bank(temp_tags):
    new_list = []
    for w1 in labels:
        for w2 in range(len(temp_tags)):
            if same_word(w1,temp_tags[w2]):
                temp_tags[w2] = w1
    return temp_tags

def suggest(*article):
    title = rmv_punc(rmv_stopword(article[0]))
    text = rmv_punc(rmv_stopword(article[1]))
    temp = ""
    # extracting tags from title
    if title.strip() != "":
        title_words = [w for w in title.split()]
        content_words = [w for w in text.split()]
        x = list()
        for t in range(len(title_words)):
            for c in range(len(content_words)):
                if title_words[t] == content_words[c]:
                    x.append(title_words[t])
                if same_word(title_words[t],content_words[c]):
                    if(len(title_words[t])<=len(content_words[c])):
                        x.append(title_words[t])
                    else:
                        x.append(title_words[t])

        counter = Counter(x)
        tag_candidates = [w[0] for w in counter.most_common(3) if w[1]>=3]
        temp = match_w_tag_bank(tag_candidates)


    # predict tags from trained model
    maxlen = 200
    text = rmv_punc(text)
    text = rmv_stopword(text)
    text = tk.texts_to_sequences([text])
    text = pad_sequences(text,maxlen=maxlen)
    text = np.array(text)
   
    pred_labels = model.predict([text])[0]
    tag_prob = dict([(labels[i], prob) for i, prob in enumerate(pred_labels.tolist())])
    pred = sorted(tag_prob.items(), key=lambda kv: kv[1],reverse=True)[:3]
    tag_proper = [w[0] for w in pred]
    l = temp + tag_proper
    s = ","
    s = s.join([w for w in l])
    return s

art1 = input("Input Title ")
art2 = input("Input Full Text ")
print(suggest(art1,art2))
