import os

import numpy as np
import pandas as pd
import torch
from pytorch_transformers import (BertConfig, BertForSequenceClassification,
                                  BertTokenizer)
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from oops.preprocess import Preprocess


class Train:
  def __init__(self, MAX_LEN:int = 150, BATCH_SIZE:int = 16):
    # SET YOUR SENTENCE LENGTH AND BATCH SIZE
    self.MAX_LEN = MAX_LEN
    self.BATCH_SIZE = BATCH_SIZE
    self.data = None
    self.preprocess = Preprocess()

    self.config = None
    self.tokenizer = None
    self.model = None
    self.optimizer = None

  def initilize_model(self):
    """
    Function to initialize the model
    """
    # BERT configuration and model initialization
    self.config = BertConfig.from_pretrained('bert-base-uncased')
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    self.model = BertForSequenceClassification(self.config)
    # FINE TUNING
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
    else:
      param_optimizer = list(self.model.classifier.named_parameters())
      optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    self.optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

  def load_data(self):
    """
    Function to load the data into a panda dataframe
    """
    self.data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),\
                          'data','text_emotion.csv'))

  def flat_accuracy(self, preds, labels):
    """
    To give the flat accuracy for the model output
    """
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

  def preprocess(self):
    """
    Function to get the required data
    """
    # I choose to train only for two labels
    # Do not use this preprocess if you want to train
    # for all the other emotions as well
    self.data = self.preprocess._Preprocess__remove_rows(self.data,['anger','boredom',
    'enthusiasm','empty','fun','relief','surprise','love','hate','neutral','worry'])
    # reset the index
    self.data.index = range(len(self.data))

  def make_sequence(self, example:str):
    """
    Makes a sequence of data based on max_length and returns
    input_id, input_mask and segment_id.
    Note: input_mask and segment_id aren't required for this task
    """
    tokens_temp = self.tokenizer.tokenize(example)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_temp) > self.MAX_LEN - 2:
        tokens_temp = tokens_temp[:(self.MAX_LEN - 2)]
    
    tokens = ["[CLS]"] + tokens_temp + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == self.MAX_LEN
    assert len(input_mask) == self.MAX_LEN
    assert len(segment_ids) == self.MAX_LEN

    return input_ids

  def make_train_test_tensor(self):
    """
    To make train and test tensors for training
    """
    input_ids = list()
    tags = list()
    for idx in range(len(self.data)):
      input_ids.append(self.make_sequence(self.data.content[idx]))
      tags.append([self.data.content[idx]])

    # SPLIT INTO TRAIN AND TEST
    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                random_state=42, test_size=0.1)

    # CONVERT TO TORCH TENSORS
    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)

    train_data = TensorDataset(tr_inputs, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.BATCH_SIZE)

    valid_data = TensorDataset(val_inputs, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.BATCH_SIZE)

    return train_dataloader,valid_dataloader

  def start_train(self):
    pass
