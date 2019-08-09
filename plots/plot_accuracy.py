import matplotlib.pyplot as plt
import pandas as pd
import os

plot_path = os.path.join(os.path.dirname(__file__),'accuracy.png')

# accuracies obtained from all the algorithms
accuracy = {'MultinomialNB_tfidf': 0.5019267822736031,
          'MultinomialNB_countvec': 0.7736030828516378,
          'SGDClassifier_tfidf': 0.5096339113680154, 
          'SGDClassifier_countvec': 0.7842003853564548, 
          'LogisticRegression_tfidf': 0.5067437379576107, 
          'LogisticRegression_countvec': 0.7822736030828517, 
          'RandomForestClassifier_tfidf': 0.5048169556840078, 
          'RandomForestClassifier_countvec': 0.7581888246628131,
          'BERT': 0.8179190751445086}

pd.DataFrame(accuracy, index=['List of Models Used']).plot(kind='bar')
plt.ylabel('Validation Accuracy in %')
plt.xticks(rotation=0)
plt.savefig(plot_path)
plt.legend(loc=2, prop={'size': 6})
plt.show()
