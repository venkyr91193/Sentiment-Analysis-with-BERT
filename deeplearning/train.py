import os
from typing import List

import numpy as np
import pandas as pd
import torch
from pytorch_transformers import (BertConfig, BertForSequenceClassification,
                                  BertTokenizer)
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from oops.preprocess import Preprocess


class Train:
  def __init__(self, filename:str, max_seq_len:int = 50, bs:int = 32, labels:List[str] = None):
    # SET YOUR SENTENCE LENGTH AND BATCH SIZE
    self.MAX_LEN = max_seq_len
    self.BATCH_SIZE = bs
    self.labels = labels
    self.label_map = {label: i for i, label in enumerate(self.labels)}
    self.filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),\
      'data',filename)
    self.data = None
    self.preprocess = Preprocess()

    self.config = None
    self.tokenizer = None
    self.model = None
    self.optimizer = None
    if not os.path.exists(self.filepath):
      print('Invalid filename')
      exit(0)

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
    # load and preprocess
    self.load_data()
    self.preprocess_data()
    # make the raw data into input tensors for train
    self.train_dataloader,self.valid_dataloader = self.make_train_test_tensor()

    # initialize cuda with torch
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.n_gpu = torch.cuda.device_count()
    if self.n_gpu > 0:
      print('Training on',torch.cuda.get_device_name())
      self.model.cuda()
    else:
      print('Training on CPU')

  def load_data(self):
    """
    Function to load the data into a panda dataframe
    """
    self.data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),\
                          'data',self.filepath))

  def flat_accuracy(self, preds, labels):
    """
    To give the flat accuracy for the model output
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

  def preprocess_data(self):
    """
    Function to get the required data
    """
    # I choose to train only for two labels
    # Do not use this preprocess if you want to train
    # for all the other emotions as well
    label_list = ['anger','boredom','enthusiasm','empty','fun',
    'relief','surprise','love','hate','neutral','worry','happiness','sadness']
    for item in self.labels:
      if item in label_list:
        label_list.remove(item)
    self.data = self.preprocess._Preprocess__remove_rows(self.data,label_list)
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
    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (self.MAX_LEN - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == self.MAX_LEN
    assert len(input_mask) == self.MAX_LEN
    assert len(segment_ids) == self.MAX_LEN

    return input_ids, input_mask, segment_ids

  def make_train_test_tensor(self):
    """
    To make train and test tensors for training
    """
    input_ids = list()
    input_mask = list()
    segment_ids = list()
    tags = list()
    for idx in range(len(self.data)):
      inp_id, inp_mask, seg_id = self.make_sequence(self.data.content[idx])
      input_ids.append(inp_id)
      input_mask.append(inp_mask)
      segment_ids.append(seg_id)
      tags.append([self.label_map[self.data.sentiment[idx]]])

    # SPLIT INTO TRAIN AND TEST
    tr_inputs, val_inputs, tr_masks, val_masks, tr_seg, val_seg,\
      tr_tags, val_tags = train_test_split(input_ids, input_mask, \
        segment_ids, tags, random_state=42, test_size=0.2)

    # CONVERT TO TORCH TENSORS
    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)
    tr_seg = torch.tensor(tr_seg)
    val_seg = torch.tensor(val_seg)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_seg, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.BATCH_SIZE)

    valid_data = TensorDataset(val_inputs, val_masks, val_seg, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.BATCH_SIZE)

    return train_dataloader,valid_dataloader

  def start_train(self, epochs:int = 5):
    """
    Function to train the model
    """
    max_grad_norm = 1.0

    for _ in trange(epochs, desc="Epoch"):
        # TRAIN loop
        self.model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(self.train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_segment_ids, b_labels = batch
            '''
            b_input_ids = b_input_ids.long()
            b_labels = b_labels.long()
            '''
            # forward pass
            output = self.model(b_input_ids, b_segment_ids, b_input_mask, labels=b_labels)
            loss = output[0]
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
            # update parameters
            self.optimizer.step()
            self.model.zero_grad()
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps))

  def start_eval(self):
    """
    Function to evaluate the model
    """
    # EVALUATING YOUR MODEL
    self.model.eval()
    predictions = []
    true_labels = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in self.valid_dataloader:
        batch = tuple(t.to(self.device) for t in batch)
        b_input_ids, b_input_mask, b_segment_ids, b_labels = batch
        '''
        b_input_ids = b_input_ids.long()
        b_labels = b_labels.long()
        '''
        with torch.no_grad():
            output = self.model(b_input_ids, b_segment_ids, b_input_mask, labels = b_labels)
            tmp_eval_loss = output[0]
            logits = output[1]
        logits = logits.detach().cpu().numpy()
        predictions.extend([p for p in np.argmax(logits, axis=1)])
        label_ids = b_labels.to('cpu').numpy()
        true_labels.extend(label_ids.flatten())
        tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    print("Validation loss: {}".format(eval_loss/nb_eval_steps))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    print("Validation F1-Score: {}".format(f1_score(predictions, true_labels,average='micro')))
    print("Classification Report")
    print(confusion_matrix(predictions, true_labels))

  def save_checkpoint(self, output_dir:str, model_name:str):
    """
    Function to save the model checkpoint at any point of the train or
    at the end.
    Saving as checkpoints allows you to load the model and update it again.
    """
    if os.path.exists(output_dir) and os.listdir(output_dir):
      print("Output directory ({}) already exists and is not empty.".format(output_dir))
      exit(0)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    PATH = os.path.join(output_dir,model_name+'.pt')
    torch.save({
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      }, PATH)
