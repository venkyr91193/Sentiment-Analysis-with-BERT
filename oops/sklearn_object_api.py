from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB


class SkLearn:
  def __init__(self):
    self.tfidf = self.tf_idf()
    self.count_vect = self.count_vectorization()
    self.label_encoder = self.get_label_encoder()
    self.obj_list = list()
    self.obj_list.append(self.multinominal_naive_base_classifier())
    self.obj_list.append(self.linear_support_vector_machine())
    self.obj_list.append(self.logistic_regression())
    self.obj_list.append(self.random_forest())

  def get_label_encoder(self):
    """
    Function to get the object of a Label Encoder
    """
    return preprocessing.LabelEncoder()

  def tf_idf(self):
    """
    Function to get the object of tf-idf
    """
    return TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))
  
  def count_vectorization(self):
    """
    Function to get the object of Count Vectorizer
    """
    return CountVectorizer(analyzer='word')

  def multinominal_naive_base_classifier(self):
    """
    Function to get the object of Multinomial Naive Bayes Classifier
    """
    return MultinomialNB()
  
  def linear_support_vector_machine(self):
    """
    Function to get the object of Linear Support Vector Machine
    """
    return SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
  
  def logistic_regression(self):
    """
    Function to get the object of Logistic Regression
    """
    return LogisticRegression(C=1)
  
  def random_forest(self):
    """
    Function to get the object of Random Forest Classifier
    """
    return RandomForestClassifier(n_estimators=500)

