import os

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from oops.preprocess import Preprocess
from oops.sklearn_object_api import SkLearn


class Algorithms:
  # making class immutable
  __slots__ = ["sklearn_obj","preprocess","data","result"]
  def __init__(self):
    # object for sklearn api
    self.sklearn_obj = SkLearn()
    self.preprocess = Preprocess()
    self.data = None
    self.result = dict()

  def load_data(self):
    """
    Function to load the data into a panda dataframe
    """
    self.data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),\
                          'data','text_emotion.csv'))

  def get_accuracy_score(self, obj, xtrain, ytrain, xval, yval):
    """
    Function to return the accuracy score of the prediction
    """
    obj.fit(xtrain, ytrain)
    y_pred = obj.predict(xval)
    return accuracy_score(y_pred, yval)

  def run_algorithms(self):
    """
    Function to run the all the Machine Learning Classifiers
    from sklearn
    """
    # load data
    self.load_data()
    # preprocess
    self.data = self.preprocess.preprocess(self.data)

    # for label encoding into integers such as 0 or 1
    labels = self.sklearn_obj.label_encoder.fit_transform(self.data.sentiment.values)

    # splitting the data into train and test using the train_test_split 
    # function from sklearn also the split is for 90% for train and 10% for validation
    # stratify just to make sure test and train have even ratio of splits
    # (not really necessary but good to validate)
    X_train, X_val, y_train, y_val = train_test_split(self.data.content.values, labels,\
       stratify=labels, random_state=42, test_size=0.1, shuffle=True)

    # apply tf-idf for the train and test and get the numbers 
    X_train_tfidf = self.sklearn_obj.tfidf.fit_transform(X_train)
    X_val_tfidf = self.sklearn_obj.tfidf.fit_transform(X_val)

    # this is for the second feature called count vectorization
    # this basically counts the number of appearances of the word in the text
    # initialize count vector
    self.sklearn_obj.count_vect.fit(self.data['content'])
    X_train_count =  self.sklearn_obj.count_vect.transform(X_train)
    X_val_count =  self.sklearn_obj.count_vect.transform(X_val)

    # getting the list of classifier objects from custom SKLEARN
    list_sklearn_objs = self.sklearn_obj.obj_list

    # implementing the classifiers
    for obj in list_sklearn_objs:
      self.result[type(obj).__name__+'_tfidf'] = self.get_accuracy_score(obj, X_train_tfidf, y_train, X_val_tfidf, y_val)
      self.result[type(obj).__name__+'_countvec'] = self.get_accuracy_score(obj, X_train_count, y_train, X_val_count, y_val)
      