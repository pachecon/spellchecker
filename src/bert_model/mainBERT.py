import sys
sys.path.insert(1, '../extras/')

import argparse
import logging
import numpy as np 
import os
import pandas as pd
import time

import spell_checker as sc
import abbreviations as abbr 

from tqdm import tqdm
from utils_bert import rewrite_file,prepare_one_file,append_results

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

def main():
    parser.add_argument("--data_dir",
                      default='../../data/misspelled_corpora/',
                      type=str,
                      required=False,
                      help="The input data dir. where the files as csv file are located.")
  
    parser.add_argument("--checkpoint_dir",
                      default="./save_ckpt/",
                      type=str,
                      required=False,
                      help="The directory to save the checkpoint or where the last checkpoint was saved.")
    
    parser.add_argument("--root_pretrained",
                      default='../pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/',
                      type=str,
                      required=False,
                      help="The directory of the pre-trained clinicalBERT")

    parser.add_argument("--dict_root",
                      default='../../dictionaries/',
                      type=str,
                      required=False,
                      help="The directory of the pre-trained clinicalBERT")
    
    parser.add_argument("--output_dir",
                      default='../../output/results/bert/',
                      type=str,
                      required=False,
                      help="The directory where the results will be saved and/or updated.")
    
    parser.add_argument("--max_num_candidates",
                      default=5,
                      type=int,
                      help="Stablish the number of candidates to be found for each misspellings.")

    args = parser.parse_args()
    fname=args.data_dir+'test_corpus.csv' # misspelled_word_test
    #Initialization of the Spell Checker Model
    spell_check = sc.SpellChecker(args.root_pretrained, args.checkpoint_dir, args.dict_root, args.output_dir,top_k=3)
    #Initialization of the Abbreviation object to modify the found abbreviations from the sentence
    check_abbr = abbr.Abbreviation(dict_root=args.dict_root)  
    
    start = 0
    for type_ in ['test']:#, 'val']:
        data_experiments(fname,spell_check,check_abbr, type_,start,args.output_dir)



def data_experiments(fname,spell_check,check_abbr, type_, start,output_dir):
    """
    Example for runing the file dending the type of the file (train, test or val)

    """
    print('Starting '+type_+' experiment..')
    originals, errors = prepare_one_file(fname,type_)
    originals = originals[start:]
    errors = errors[start:]

    errors = ['diastolic turbulence of left carofid artery']
    df = pd.DataFrame(columns=['sentences', 'suggested', 'probability','label', 'time'])
        
    for i,sentence in enumerate(tqdm(errors)):
        label = originals[i]
        sentence = check_abbr.change_abbreviation(sentence)
        df = spell_check.get_final_suggested_sentences(sentence,df, label=label)
        df = df.reset_index(drop=True)
        print(df.suggested.values, df.probability.values)
        exit()
        if (i % 10 == 0):
            #print('Loop {}, the results have been stored'.format(i))
            #print('---------------------------------------------------------------------------------------------------------------------')
            rewrite_file(df, output_dir , type_)
            df = pd.DataFrame(columns=['sentences', 'suggested', 'probability','label', 'time'])
    rewrite_file(df, output_dir ,type_)

if __name__ == "__main__":
    main()