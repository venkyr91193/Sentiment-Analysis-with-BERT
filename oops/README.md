# Explaination of the class components

# Algorithms class

This is the class to initialize the algorithms you are going to run. Follow the steps
as below:

    >>> obj = Algorithms()
    >>> obj.run_algorithms()
    >>> obj.result

The obj.result will provide you a dictionary with all the sklearn classifier model's validation score 
which is trained on two types of inputs like count vectorization or tf-idf.

# Preprocess class

This is the class to preprocess your data that comes in similar to that of the tutorial except 
its a class implementation which is called internally in the algorithms.py code as shown below:

    >>> preprocess = Preprocess()
    >>> data = preprocess.preprocess(data)

# SkLearn class

This is the class to initialize all the sklearn classifiers modules.
Since sklearn classifiers work on fit() and predict(), its easier to use them from a 
class object since everything has a common implementation. Here is how you use it.

    >>> sklearn_obj = SkLearn()
    >>> sklearn_obj.obj_list

The obj_list parameter of the SkLearn class returns the list of sklearn classifier objects initialized and ready to use.
Also the tf-idf and count vectorization can also be called as follows:

sklearn_obj.tfidf -> gives the tf-idf object.

sklearn_obj.count_vect -> gives the count vectorization object.

