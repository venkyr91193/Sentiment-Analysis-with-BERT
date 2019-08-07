import os

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("ggplot")

class DataAnalyser:
  def __init__(self):
    self.data = None

  def load_data(self):
    """
    Function to load the data into a panda dataframe
    """
    self.data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),\
                          'data','text_emotion.csv'))
  
  def analyse(self):
    """
    Function to analyse the given data
    """
    self.load_data()

    # length list of sentences
    length_sents = list()

    for idx in range(len(self.data)):
      sent = self.data.content[idx]
      length_sents.append(len(sent))
    
    print('Max length of the sentences present',max(length_sents))
    plt.hist(length_sents,bins=50)
    plt.show()
