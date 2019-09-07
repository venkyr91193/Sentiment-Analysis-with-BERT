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
This component is used to train your model based on your requirements. The input file should be a csv file.
Please work from the root directory of the folder.

Data should be in csv in the folder .\data with the required columns 'sentiment' and 'content' 
which has your emotions and text sentences respectively.

    >>> from deeplearning.train import Train
    >>> obj = Train('yourfilename.csv',max_seq_len=40,bs=8,labels=['happiness','sadness'])
    # object initilization fails if the filename is incorrect
    # you can also set the seq length,batch size and labels you want to train explicitly
    >>> obj.max_seq_len = 50
    >>> obj.bs = 32
    >>> obj.labels = ['happiness','sadness']
    >>> obj.initilize_model()
    >>> obj.start_train(epochs=2)
    >>> obj.start_eval()

# Save the model
To save the model at any given checkpoint or training. You can use as shown below.

    >>> obj.save_checkpoint('your_output_model_path','model_name')

# Load the model
To load the model again for training or testing. Here is how you do it.

    >>> from pytorch_transformers import BertConfig, BertForSequenceClassification
    >>> config = BertConfig()
    >>> model = BertForSequenceClassification(config)
    >>> checkpoint = torch.load('your_output_model_path\model_name.pt')
    >>> checkpoint.keys()
    dict_keys(['model_state_dict', 'optimizer_state_dict'])
    >>> model.load_state_dict(checkpoint['model_state_dict'])
    IncompatibleKeys(missing_keys=[], unexpected_keys=[])
    >>> model.eval()

I have fixed the max sequence length to 50 taking into account start and end tokens after 
observing the result from the DataAnalyzer class which gave me results giving information 
of 40 words per sentence (shown in figure analysed.png in \plots folder). Its adviced to 
set the batch size to 64 according to the BERT papers. But you can decrease it if your GPU 
memory is low. I have defaulted the batch size to 32.

# Results

With just 2 epochs I was able to achieve 82% accuracy on validation data as shown.
    Training on GeForce GTX 1060
    Validation loss: 0.4211259511384097
    Validation Accuracy: 0.8185876623376623
    Validation F1-Score: 0.8179190751445086

One epoch toook 2.2 minutes on an average training on GTX 1060.

