import csv
import dill
import json
import pandas as pd 
import re
import sys

#import fnvhash
#import hash_ngrams
#import searching
def rewrite_file(df, type_exp, output_dir='../../output/results/baseline/'):
    """
    Save the results in the output file
    Parameters
    ----------
    df:pandas.DataFrame
        data to be saved or rewrite as csv file
    type_exp:str
        labels for train, test or val
    output_dir:str
        directory of the output folder where the results are saved
    """
    df.to_csv(output_dir+'noisychannel_out_'+type_exp+'.csv',mode ='a', index = False, header=False)

def tokenize_file(file) :
    """
    Read the file, tokenize and build a list of sentences
    """
    data = pd.read_csv(file)
    data = data.dropna()
    sentences = []
    for i,sentence in enumerate(data.words): #tokenizer.tokenize(content):
        print("\rFold {}/{}.".format(i, len(data.words)), end='\r')
        sys.stdout.flush()
        sentence_clean = [i.lower() for i in re.split('[^a-zA-Z]+', sentence) if i]
        sentences.append(sentence_clean)
    return sentences

def save_total(OUT_PATH_TRAIN,obj, name):
    with open(OUT_PATH_TRAIN+'baseline_ngrams/'+ name + '.pkl', 'wb') as f:#'wb' 
        dill.dump(obj,f)

def save_ngrams(OUT_PATH_TRAIN,obj, name):
    with open(OUT_PATH_TRAIN+'baseline_ngrams/'+ name + '.csv', 'w') as f:
        writer = csv.DictWriter(f,fieldnames=['key','value'])
        for i,row in enumerate(obj):
            print("\rFold {}/{}.".format(i, len(obj)), end='\r')
            sys.stdout.flush()
            writer.writerow({'key':row,'value':obj[row]})

def __save_ngrams_json__(OUT_PATH_TRAIN,obj, name):
    with open(OUT_PATH_TRAIN+'baseline_ngrams/'+ name + '.json', 'w') as f:
        writer = csv.DictWriter(f,fieldnames=['key','value'])
        json.dump(obj,writer, sort_keys=True)

def load_pickle(OUT_PATH_TRAIN,name):
    with open(OUT_PATH_TRAIN+'baseline_ngrams/' + name + '.pkl', 'rb') as f:
        return dill.load(f)

def load_ngram(OUT_PATH_TRAIN,name):
    return pd.read_csv(OUT_PATH_TRAIN+'baseline_ngrams/' + name+'_order.csv')
    
def __hashing_func__(key):
    return fnvhash.fnv1a_32(key) #hash(key) % len_table

def get_ngrams_hash(OUT_PATH_TRAIN, bigram_key):
    df = pd.read_csv(OUT_PATH_TRAIN+'baseline_ngrams/bigrams_hashed_fnv.csv')#bigrams_hashed_name
    hash_key = __hashing_func__(str.encode(bigram_key))
    value = df.loc[df.key == hash_key,['value']].values
    if not value:
        return None
    if value.ndim == 1:
        value_1 =value[0]
    elif value.ndim == 2:
        value_1 = float(value[0][0])
    return value_1

def get_value_binary(bigram_key):
    return searching.binary_search(bigram_key)

def clean_word(text):
    text = text.lower() 
    text = re.sub(r'\"+', '', text)
    text = re.sub(r'\'+', '', text)
    text = text.strip()
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'[{}@_*();:\\#%+=\-\[\]\.\,\?\!]+',' ', text)
    text = re.sub(r'a0',r'', text)
    text = re.sub(r'\'92t',r'\'t', text)
    text = re.sub(r'\'92s',r'\'s', text)
    text = re.sub(r'\'92m',r'\'m', text)
    text = re.sub(r'\'92ll',r'\'ll', text)
    text = re.sub(r'[\b\ufeff\b]',r'',text)
    #text = re.sub(r'[0-9]+',r'',text)
    return text
