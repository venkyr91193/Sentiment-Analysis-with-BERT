# Explaination of the plots

As you can see from the accuracy chart from (accuracy.png), it clearly states that BERT
is superior to other algorithms. With only 1 iteration we can train the model to perform 
very well on a particular language because of its implementation of its multiple attention 
heads. This is from the deep learning side.

On the machine learning side we can see the results perform better with count vectorization which
is a simple count of all the occurances of the words present in the sentence. From this we can say that
a single word like 'happy' or 'joy' has a major effect on its sentiment. But in sentences like,
'all the people around me are happy, but I am sad' will definitely give a wrong prediction if the words alone
are being used to judge a sentiment. This is the place where training a model with attention heads comes into
picture. Attention heads in short map the word I to sad giving sad the more priority than the happy (more complex
than its written here). Therefore training a sentence in both the directions with its context is the best way.

