import re
from typing import List

import pandas as pd
from nltk.corpus import stopwords
from pandas import DataFrame
from textblob import Word


class Preprocess:
  def __init__(self):
    pass

  @staticmethod
  def __remove_repeat(text:str):
    '''
    Function to filter out the letter repetitions
    '''
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

  def preprocess(self, data:DataFrame)->DataFrame:
    """
    Preprocessing the input data
    """
    # dropping the column author because its not required
    data = self.__remove_columns(data,['author'])
    # keeping only rows with 'happiness' and 'sadness' labels
    list_to_remove = ['anger','boredom','enthusiasm','empty',
    'fun','relief','surprise','love','hate','neutral','worry']
    data = self.__remove_rows(data,list_to_remove)
    data.index = range(len(data))
    data = self.__lower_strip_symbols(data)
    data = self.__remove_stop_words(data)
    daat = self.__match_lemma(data)

    # removing the repetitions in the letters
    data['content'] = data['content'].apply(lambda x: " ".join(Preprocess.__remove_repeat(x) for x in x.split()))

    data = self.__remove_rare_words(data)
    return data

  def __remove_columns(self, data:DataFrame, columns:List[str])->DataFrame:
    """
    Function to drop the unnecessary columns
    """
    for column in columns:
      data = data.drop(column, axis=1)
    return data
  
  def __remove_rows(self, data:DataFrame, rows:List[str])->DataFrame:
    """
    Function to drop the unnecessary rows
    """
    for row in rows:
      data = data.drop(data[data.sentiment == row].index)
    return data

  def __lower_strip_symbols(self, data:DataFrame)->DataFrame:
    """
    Function to remove the symbols, lower the string case
    and remove extra endlines char.
    """
    # convert the string to lowercase
    data['content'] = data['content'].str.lower()
    # removing the end of line as well
    data['content'] = data['content'].str.rstrip()
    # removing punctuation, symbols 
    data['content'] = data['content'].str.replace('[^\w\s]',' ')
    return data

  def __remove_stop_words(self, data:DataFrame)->DataFrame:
    """
    Function to remove the stop words from the content
    """
    # removing the stop words using NLTK. Can be done with spacy as well
    stop = stopwords.words('english')
    data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    return data

  def __match_lemma(self, data:DataFrame)->DataFrame:
    """
    Convert all the words to its base word or lemma
    """
    # making all the words to match the vocabulary using its lemma
    data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return data

  def __remove_rare_words(self, data:DataFrame)->DataFrame:
    """
    Function to remove the rare words or less frequent words
    """
    # removing the less used words which is not going to have impact on the model
    # join all the texts into one single texts and generate the value count and
    # taking only top rarest to remove from the content in the coming lines
    freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]
    # removing the rare words in the freq list from the content column
    freq = list(freq.index)
    data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    return data
