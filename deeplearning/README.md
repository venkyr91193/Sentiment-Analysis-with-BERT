# Explaination of the class components

# DataAnalyser component
This component is used to analyse the input data to find the maximum sentence length and
it plots an histogram to view the results for best selection of the "max_seq_len" parameter
of the Train class component.

To use this API, follow the steps as below:

    >>> from deeplearning.data_analyser import DataAnalyser
    >>> obj = DataAnalyser()
    >>> obj.analyse()

# Train component
This component is used to train your model based on your requirements

    >>> obj = Train(max_seq_len=150,bs=8,labels=['happiness','sadness'])
    # you can also set the seq length,batch size and labels you want to train explicitly
    >>> obj.max_seq_len = 75
    >>> obj.bs = 8
    >>> obj.labels = ['happiness','sadness']
    >>> obj.initilize_model()
    >>> obj.start_train(epochs=2)
    >>> obj.start_eval()