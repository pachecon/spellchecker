import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
This file is used only during training and validation for the BERT_GED_MODEL
It is for preparing all csv files into one big column of lables and input_sentences
where labels are the correct sentence without any type of misspells
and sentences are the ones with any type of misspells.

"""
def __load_data__(fname, num_rows=50000, colab=True, limit_data=False):
  if colab:
    df = pd.read_csv (fname, delimiter='\t', header=None, 
                  names=['sentence_source', 'label', 'label_notes', 'sentence'])
    return df
  else:
    df_med = pd.read_csv(fname)
    if limit_data:
      df_med = df_med[:num_rows]
    df_med = df_med.dropna().reset_index(drop=True)
    df_med = df_med.sample(frac=1).reset_index(drop=True)
    return df_med

def __append_onsame__(df2):
  df1 = pd.DataFrame(columns=['sentence', 'label'])
  df1['sentence'] = df2.original.append(df2.errors,ignore_index=True)#df
  df1['label'] =  df2['label'].append(pd.Series(np.zeros(len(df2.errors))),ignore_index=True) # np.ones(len(df))
  return df1

def __append_datasets__(df2,df3, errors = False):
  df1 = pd.DataFrame(columns=['sentence', 'label'])
  df1['sentence'] = df3.sentence.append(df2,ignore_index=True)#df
  if errors:
    df1['label'] =  df3['label'].append(pd.Series(np.zeros(len(df2))),ignore_index=True) # np.ones(len(df))
  else:
    df1['label'] =  df3['label'].append(pd.Series(np.ones(len(df2))),ignore_index=True) # np.ones(len(df))
  return df1

def __prepare_sentences_labels__(df, df_train):
  print('Prepare sentences and labels for training...')
  # Sentence and Label Lists
  sentences = df.sentence.values
  sentences = np.append(sentences, df_train.sentence.values)
  
  labels = df.label.values
  labels = np.append(labels, [int(v) for v in df_train.label.values])
  del df, df_train
  assert(len(labels) == len(sentences))
  return sentences, labels

def prepare_data_train():
  train_data = '../../data/misspelled_corpora/misspelled_word_non_train.csv'
  more_data_train = '../../data/misspelled_corpora/misspelled_word_error_train.csv'
  colab_data = '../../data/train/cola_public/raw/train.tsv'
  
  df = __load_data__(colab_data)
  df_med = __load_data__(train_data, colab=False, limit_data=True)
  df_med['label'] = pd.Series(np.ones(len(df_med.original)))
  df_new = __append_onsame__(df_med)

  df_med = __load_data__(more_data_train, colab=False, limit_data=True)
  df_med_correct = pd.Series([c[0] for c in df_med.original])
  df_med_incorrect = pd.Series([c for x in df_med.errors for c in x.split(sep=',')])
  del df_med
  
  df_dummy = __append_datasets__(df_med_correct,df_new, errors = False)
  df_train = __append_datasets__(df_med_incorrect, df_dummy, errors = True)
  
  del df_dummy, df_new, df_med_correct, df_med_incorrect
  sentences, labels = __prepare_sentences_labels__(df, df_train)
  del df, df_train
  return sentences, labels 

def prepare_data_val():
  val_data = '../../data/misspelled_corpora/misspelled_word_non_val.csv'
  more_data_val = '../../data/misspelled_corpora/misspelled_word_error_val.csv'
  
  df_med = __load_data__(val_data,num_rows=200, colab=False, limit_data=True)
  df_med['label'] = pd.Series(np.ones(len(df_med.original)))
  df_new = __append_onsame__(df_med)

  df_med = __load_data__(more_data_val,num_rows=200, colab=False, limit_data=True)
  df_med_correct = pd.Series([c[0] for c in df_med.original])
  df_med_incorrect = pd.Series([c for x in df_med.errors for c in x.split(sep=',')])
  del df_med
  
  df_dummy = __append_datasets__(df_med_correct,df_new, errors = False)
  df_train = __append_datasets__(df_med_incorrect, df_dummy, errors = True)
  del df_dummy, df_new, df_med_correct, df_med_incorrect
  sentences = df_train.sentence.values 
  labels = [int(v) for v in df_train.label.values]
  del df_train
  assert(len(labels) == len(sentences))
  return sentences, labels 

def prepare_data_train_val(fname):
    df = __load_data__(fname, colab=False)
    columns_name =df.columns
    return df[columns_name[1]], df[columns_name[0]] #sentences, labels