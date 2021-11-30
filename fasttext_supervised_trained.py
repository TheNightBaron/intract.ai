import fasttext '''facebook implementation of fasttext since gensim doesnt support
supervised learning fasttext yet'''
help(fasttext.FastText)

# Text Classification with fastText
# Importing libraries
import numpy as np, pandas as pd
import csv
# NLP Preprocessing
from gensim.utils import simple_preprocess

import string

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
colors = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(colors))
matplotlib.rcParams['figure.figsize'] = 12, 8

'''importing twitter dataset created by me'''
t_data = pd.read_excel(r'/Users/harshgarg/Desktop/intract/Text Analysis/TwitterAPI/data.xlsx')
#Taking only the cleaned text and labels
emotion = {'joy':'joy',
           'excited':'joy',
           'happy':'joy',
           'fear':'fear',
           'scared':'fear',
           'afraid':'fear',
           'anticipation':'anticipation',
           'hope':'anticipation',
           'trust':'trust',
           'pride':'pride'}
t_data['emotion'] = t_data['Keyword'].apply(lambda x: emotion.get(x))
t_data = t_data[['cleaned_text','emotion']]

######Get twitter data sourced from following TDS article
# https://towardsdatascience.com/analysis-of-the-emotion-data-a-dataset-for-emotion-recognition-tasks-6b8c9a5dfe57

'''option 1: through wget'''

# !wget https://www.dropbox.com/s/607ptdakxuh5i4s/merged_training.pkl

# Defining a helper function to load the data
import pickle

def load_from_pickle(directory):
    return pickle.load(open(directory,"rb"))

# Loading the data
data = load_from_pickle(directory="merged_training.pkl").reset_index(drop = True)
data.head()

'''option 2: using huggingface datasets'''

from datasets import load_dataset

emotion_dataset = load_dataset("emotion")
emotion_dataset
emotion_train = emotion_dataset['train']
print(emotion_train[0])
print(emotion_train.column_names)
print(emotion_train.features)

emotion_dataset.set_format(type="pandas")
train = emotion_dataset["train"][:]
test = emotion_dataset["test"][:]
val = emotion_dataset["validation"][:]

labels_dict = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
required_labels = ['joy','fear','pride','trust','anticipation']

train['description'] = train['label'].map(labels_dict )
test['description'] = test['label'].map(labels_dict )
val['description'] = val['label'].map(labels_dict )

train = train[train['description'].isin(required_labels)]
test = test[test['description'].isin(required_labels)]
val = val[val['description'].isin(required_labels)]

train['description'].value_counts(normalize=True)

'''distribution of labels in the training set'''
sns.countplot(train['description'],order = train['description'].value_counts(normalize=True).index)
train['text_length'] = train['text'].astype(str).apply(len)
train['text_word_count'] = train['text'].apply(lambda x: len(str(x).split()))

'''tweet length analysis'''
sns.distplot(train['text_length'])
plt.xlim([0, 512]);
plt.xlabel('Text Length');

'''tweet word count per label'''
sns.boxplot(x="description", y="text_word_count", data=train)

'''end of data importing and EDA'''

def processing(df, labels = 'description', text_cols = 'text'):
    df = df[[text_cols, labels]]
    df[text_cols] = df[text_cols].apply(lambda x: ' '.join(simple_preprocess(x)))
    df[labels] = df[labels].apply(lambda x: '__label__' + x)
    df.columns = ['text','description']
    return df

t_data = processing(t_data, 'emotion', 'cleaned_text')
train = processing(train)
test = processing(test)
val = processing(val)

train = pd.concat([t_data, train])

train[['description', 'text']].to_csv('train.txt', 
                                          index = False, 
                                          sep = ' ',
                                          header = None, 
                                          quoting = csv.QUOTE_NONE, 
                                          quotechar = "", 
                                          escapechar = " ")
test[['description', 'text']].to_csv('test.txt', 
                                          index = False, 
                                          sep = ' ',
                                          header = None, 
                                          quoting = csv.QUOTE_NONE, 
                                          quotechar = "", 
                                          escapechar = " ")
val[['description', 'text']].to_csv('val.txt', 
                                          index = False, 
                                          sep = ' ',
                                          header = None, 
                                          quoting = csv.QUOTE_NONE, 
                                          quotechar = "", 
                                          escapechar = " ")

# Training the fastText classifier
model = fasttext.train_supervised('train.txt', wordNgrams = 3)

# Evaluating performance on the entire test file
model.test('val.txt',k=1)
'''(Number of rows, precision at k, recall at k)'''

# Predicting on a single input
model.predict(test.iloc[2, 0])

# Save the trained model
#model.save_model('model.bin')




