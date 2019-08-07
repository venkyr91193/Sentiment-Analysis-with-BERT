# Explaination of the class components

# DataAnalyser component
This component is used to analyse the input data to find the maximum sentence length and
it plots an histogram to view the results for best selection of the "max_seq_len" parameter
of the Train class component.

To use this API, follow the steps as below:

    >>> from deeplearning.data_analyser import DataAnalyser
    >>> obj = DataAnalyser()
    >>> obj.analyse()

