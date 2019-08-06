import os
import re
import sys

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from textblob import Word

def remove_repeat(text):
  '''
  Function to filter out the letter repetitions
  '''
  pattern = re.compile(r"(.)\1{2,}")
  return pattern.sub(r"\1\1", text)

'''
LOADING THE DATA
'''
# loading the file
data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),'data','text_emotion.csv'))


'''
PREPROCESSING OF THE DATA
'''
# dropping the column author because its not required
data = data.drop('author', axis=1)
# keeping only rows with 'happiness' and 'sadness' labels
data = data.loc[(data["sentiment"] == 'happiness') | (data['sentiment'] == 'sadness')]
# reset the index
data.index = range(len(data))
# convert the string to lowercase
data['content'] = data['content'].str.lower()
# removing the end of line as well
data['content'] = data['content'].str.rstrip()
# removing punctuation, symbols 
data['content'] = data['content'].str.replace('[^\w\s]',' ')
# removing the stop words using NLTK. Can be done with spacy as well
stop = stopwords.words('english')
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
# making all the words to match the vocabulary using its lemma
data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
# removing the repetitions in the letters
data['content'] = data['content'].apply(lambda x: " ".join(remove_repeat(x) for x in x.split()))

# removing the less used words which is not going to have impact on the model
# join all the texts into one single texts and generate the value count 
# taking only top rarest to remove from the content in the coming lines
freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]

# removing the rare words in the freq list
freq = list(freq.index)
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

'''
ENCODING OF THE LABELS
'''
# for label encoding into integers such as 0 or 1
label_encoder = preprocessing.LabelEncoder()
labels = label_encoder.fit_transform(data.sentiment.values)

'''
TRAIN TEST SPLIT
'''
# splitting the data into train and test using the train_test_split function from sklearn
# also the split is for 90% for train and 10% for validation
# stratify just to make sure test and train have even ratio of splits(not really necessary but good to validate)
X_train, X_val, y_train, y_val = train_test_split(data.content.values, labels, stratify=labels, random_state=42, test_size=0.1, shuffle=True)

'''
TF-IDF CALCULATIONS
'''
# intialize tf-idf
tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))
# apply tf-idf for the train and test and get the numbers 
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.fit_transform(X_val)

'''
COUNT VECTOR CALCULATIONS
'''
# this is for the second feature called count vectorization
# this basically counts the number of appearances of the word in the text
# initialize count vector
count_vect = CountVectorizer(analyzer='word')
count_vect.fit(data['content'])
X_train_count =  count_vect.transform(X_train)
X_val_count =  count_vect.transform(X_val)


'''
ALGORITHMS
'''
# Using the machine learning models

# Model 1: Multinomial Naive Bayes Classifier
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred = nb.predict(X_val_tfidf)
print('Naive Bayes with tfidf %s' % accuracy_score(y_pred, y_val))
nb.fit(X_train_count, y_train)
y_pred = nb.predict(X_val_count)
print('Naive Bayes with count vectors %s' % accuracy_score(y_pred, y_val))

# Model 2: Linear SVM
lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
lsvm.fit(X_train_tfidf, y_train)
y_pred = lsvm.predict(X_val_tfidf)
print('Support Vector Machine with tfidf %s' % accuracy_score(y_pred, y_val))
lsvm.fit(X_train_count, y_train)
y_pred = lsvm.predict(X_val_count)
print('Support Vector Machine with count vectors %s' % accuracy_score(y_pred, y_val))

# Model 3: logistic regression
logreg = LogisticRegression(C=1)
logreg.fit(X_train_tfidf, y_train)
y_pred = logreg.predict(X_val_tfidf)
print('Logistic Regression with tfidf %s' % accuracy_score(y_pred, y_val))
logreg.fit(X_train_count, y_train)
y_pred = logreg.predict(X_val_count)
print('Logistic Regression with count vectors %s' % accuracy_score(y_pred, y_val))

# Model 4: Random Forest Classifier
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train_tfidf, y_train)
y_pred = rf.predict(X_val_tfidf)
print('Random Forest with tfidf %s' % accuracy_score(y_pred, y_val))
rf.fit(X_train_count, y_train)
y_pred = rf.predict(X_val_count)
print('Random Forest with count vectors %s' % accuracy_score(y_pred, y_val))
