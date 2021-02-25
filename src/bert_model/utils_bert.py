import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
import time
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSequenceClassification

torch.manual_seed(42) #random seed for initialization; Deafult:42

def load_model_detection(root_pretrained, path_checkpoints):
    print('Load pretrained model... ')
    model = AutoModelForSequenceClassification.from_pretrained(root_pretrained)#, from_pt=True)
    ckpt = torch.load(path_checkpoints + "model_3.pt") #2.pt
    model.load_state_dict(ckpt['model_state_dict'])
    #model.load_state_dict(torch.load(path_checkpoints + "bert-GD.pth"), strict=False)
    model.eval()
    return model

def __find_max_length__(list_tokens): 
  """
  Verify if the maximum length of the list of tokens is less than the MAX_LEN
  """
  #maxList = max(list_tokens, key = len) 
  maxLength = max(map(len, list_tokens)) 
  return maxLength 
  
def tokenize_data(tokenizer, sentences):
  """
    tokenizer: transformers.AutoTokenizer 
    sentences: List of input sentences

    1) Tokenize each input sentence
    2) Converts tokens to ids for each input sentence
    3) Add for each input sentence the special tokens: "[CLS]" and "[SEP]"

    returns for each input sentence the tokens plus the special tokens.
  """
  #print('Tokenized the input and adding special tokens... ')
  tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
  tokenized_texts = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_texts]
  tokenized_texts = [tokenizer.build_inputs_with_special_tokens(t) for t in tokenized_texts]
  del sentences
  #print("Tokenize version of the the first sentence:")
  #print(tokenized_texts[0])
  return tokenized_texts

def add_padding(tokenized_texts,MAX_LEN = 128):
  """
  Padding of input sentences, where the maximum len by default is 128.
  However, by using the method "__find_max_length__", the maxlen can be
  reduced after verifying the longest length from the dataset. 
  If the length of the data is lower than 128 (default), then the length 
  of the data is used. So, it will be better for the memory.  

  return inputs ids with padding 
  """
  #print('Prepare batch data...')
  maxLength = __find_max_length__(tokenized_texts)
  maxlen=MAX_LEN
  if maxLength < MAX_LEN:
    maxlen = maxLength
  #print('Maximum length found for this dataset: {}'.format(maxlen))
  input_ids = pad_sequences(tokenized_texts, maxlen=maxlen, 
                            dtype ="long", truncating="post",padding ="post")
  del tokenized_texts
  return  input_ids

def prepare_dataloader(tokenizer, sentences, labels, BATCH_SIZE=32, train=True):
  """
  - Prepare tokenize input sentence.
  - Index Numbers and Padding each tokenized sentence
  - Create an iterator of our data with torch DataLoader
    This helps save on memory during training
  - During training the data is randomly shuffle
  """
  #print('Load tokenizer from pretrained model...')
  tokenized_texts = tokenize_data(tokenizer, sentences)
  del sentences
  assert (len(tokenized_texts) == len(labels))
  input_ids = add_padding(tokenized_texts)
  #print('Prepare inputs for BERT inputs model...')
  input_ids =  torch.tensor(input_ids)
  labels = torch.tensor(labels)
  t_data = TensorDataset(input_ids,labels)
  if train:
    t_sampler = RandomSampler(t_data)
    data_loader = DataLoader(t_data, sampler=t_sampler, batch_size=BATCH_SIZE)
  else:
    data_loader = DataLoader(t_data, batch_size=BATCH_SIZE) #or 8?
  return data_loader

def prepare_inputs_dataloader(tokenizer, sentences, BATCH_SIZE=32):
  """
  - Prepare tokenize input sentence.
  - Index Numbers and Padding each tokenized sentence
  - Create an iterator of our data with torch DataLoader
  """
  tokenized_texts = tokenize_data(tokenizer, sentences)
  input_ids = add_padding(tokenized_texts)
  #print('Prepare inputs for BERT inputs model...')
  
  input_ids =  torch.tensor(input_ids)
  t_data = TensorDataset(input_ids)
  t_sampler = RandomSampler(t_data)
  data_loader = DataLoader(t_data, sampler=t_sampler, batch_size=BATCH_SIZE)
  return data_loader

def load_last_saved_checkpoint(path_checkpoints, epoch, model, optimizer):
  """
    Loads the last checkpoint from the last time the model was training.
    path_checkpoints: path where the checkpoints are saved
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    returns the model with the last information of the weigths, the epoch, the value of the loss,
          the batch step and the loaded optimizer
  """
  print("Loading  last checkpoint from '{}'".format("model_"+str(epoch)+".pt"))
  ckpt = torch.load(path_checkpoints + "model_"+str(epoch)+".pt")
  model.load_state_dict(ckpt['model_state_dict'])
  epoch = ckpt['epoch'] #+ 1
  loss_saved = ckpt['loss']
  steps = ckpt['step']
  optimizer.load_state_dict(ckpt['optimizer_state_dict'])
  print("Loaded checkpoint '{}' (epoch {})"
                  .format("model_"+str(epoch)+".pt", epoch))
  return model, epoch, loss_saved, steps, optimizer

def create_attention_mask(input_ids):
  """
  Create a mask of 1s for each token followed by 0s for padding for BERT input
  return tensor of attention masks
  """
  attention_masks = []
  for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)
  del input_ids
  return torch.tensor(attention_masks)

def flat_accuracy(predictions, labels):
  """
  Calculates the accuracy of our predictions vs labels
  """
  pred_flat  = np.argmax(predictions , axis=1).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat)/len(labels_flat)

def plot_train_val(train_loss_set):
  plt.figure(figsize=(15,8))
  plt.title("Training loss")
  plt.xlabel("Batch")
  plt.ylabel("Loss")
  plt.plot(train_loss_set)
  plt.show()

def save_checkpoints(epoch, model, optimizer, train_loss, step, checkpoint_dir):
  """
    Save the checkpoints during training
  """
  torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': train_loss,
              'step':step,
              }, checkpoint_dir+'model_'+str(epoch)+".pt")
  #print('Saved after 100 mini-batches')

def rewrite_file(df,out_dir, type_exp):
  """
  Appending new results into the csv file called 
  'bert_out_'
  This method is used during the testing. 
  """
  df.to_csv(out_dir+'bert_out_'+type_exp+'.csv',mode ='a', index = False, header=False)

def delete_nans(data):
  """
  Delete rows where NaNs values are found in the data set.
  """
  data = data.dropna().reset_index(drop=True)
  #data = data.sample(frac=1).reset_index(drop=True)
  return data

def clean_sentence(text):
  """
  Delete any website, extra spaces, and symbols 
  """
  text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) #remove HTML
  text = re.sub(r'\"+', '', text)
  text = re.sub(r'\'+', '', text)
  text = re.sub(r'\n', ' ', text) 
  #text = re.sub(r'[{}@_*()\\#%+=-\[\]\.\,\?\!]+',' ', text)
  text = re.sub(r'[{}@_*();:\\#%=\-\[\]\.\,\?\!]+',' ', text)
  text = re.sub(r'[\b\ufeff\b]',r'',text)
  #text = re.sub(r'[0-9]+',r'',text)
  return text

def __check_time__(start):
  """
  Calculation the computation time
  Parameter
  ---------
  start:float
      the start point when the time begins to run
  Return
  ------
  float
      the total time in seconds
  """
  end = time.time()
  return end-start

def append_results(final_output,final_dict, start_time):
  final_time = __check_time__(start_time)
  final_dict['time']=final_time
  final_out = final_output.append(final_dict, ignore_index=True)
  return final_out

def prepare_one_file(fname,type_):
  data = pd.read_csv(fname)
  data = delete_nans(data)
  return data.original.values, data.errors.values
