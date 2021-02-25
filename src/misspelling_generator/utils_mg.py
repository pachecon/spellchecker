from __future__ import absolute_import, division, print_function, unicode_literals

import os
import random
import re
import pandas as pd

from random import sample, choice 

def rewrite_file(df,output_root,output_fname):
    """
    Rewrite the output file to save new misspelled sentences.
    Parameters
    ----------
    df:
        object to be rewrite
    """
    if(os.path.exists(output_root+'misspelled_word_'+output_fname+'.csv')):
        df_temp = pd.read_csv(output_root+'misspelled_word_'+output_fname+'.csv')
        df = df.append(df_temp)
    df.to_csv(output_root+'misspelled_word_'+output_fname+'.csv', index=False)
        
def read_file(fname):
    """
    Read the input file for test, train and validation
    Return
    -------
        the data 
    """
    data = pd.read_csv(fname)#, names=['words'])
    data = data.dropna()
    data = data['words'].values
    return data

def clean_word(text):
    text = text.lower() 
    text = text.strip()
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'[{}@_*()\\#%+=\[\]\.\,\?\!\-]+',' ', text)
    text = re.sub(r'a0',r' ', text)
    text = re.sub(r'\'92t',r'\'t', text)
    text = re.sub(r'\'92s',r'\'s', text)
    text = re.sub(r'\'92m',r'\'m', text)
    text = re.sub(r'\'92ll',r'\'ll', text)
    text = re.sub(r'[\b\ufeff\b]',r' ',text)
    text = re.sub(r'\"+', '', text)
    text = re.sub(r'\'+', '', text)
    #text = re.sub(r'[0-9]+',r'',text)
    return text

def prepare_indexes_random(line,k=2):
    """
    Randomly select an index from the sentence (line)
    Parameters
    ----------
    line:str
        initial sentence
    k (Optional):int
        Default 2. It is used for indicating the number of desired words to be modified 
    Return
    ------
    array:
        n index value. This indicates the position of the word which will be modified 
    """
    j=0
    r = len(line) // k
    n = []
    while j < r:
        if random.randint(0,len(line)-1) not in n:
            n.append(random.randint(0,len(line)-1))
            j +=1 
    return n
    
def create_misspellings_file(dic_new, original, line,probs=[]):
    if probs:
        dic_new = dic_new.append({'original':original, 'errors':line, 'probabilities':probs}, ignore_index = True) #, 'phonetic':phonetic_sentence

    else:     
        dic_new = dic_new.append({'original':original, 'errors':line}, ignore_index = True) #, 'phonetic':phonetic_sentence
    return dic_new