import sys
sys.path.insert(1, './baseline/')
sys.path.insert(1, './bert_model/')
sys.path.insert(1, './prepare_data/')
sys.path.insert(1, './extras/')
sys.path.insert(1, './candidates/')

import argparse
import io
import logging

import numpy as np
import pandas as pd
import re

import abbreviations as abbr 
import detect_type as dt
import noisy_channel_model as ns
import spell_checker as sc

from utils_koehler import rewrite_file, prepare_list_files, clean_test_file, open_koehler_data, type_file
from tqdm import tqdm

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

def check_terms(check_type,bestSentence):
    words = re.split(r';|,|\n|(\band\b)|\+',bestSentence) #\*|
    #print(words)
    if None in words:
        words = list(filter(None, words)) 
    if 'and' in words:
        words.remove('and')
    word_l = ''
    terms = []
    if len(words)>1:
        word_l = [w.split() for w in words]
        #print(word_l)
        terms = check_type.search_terms(word_l)
        #print('Terms found: {}'.format([' '.join(t)for t in terms]))
    elif not words:
        terms = ''
    else:
        terms = check_type.search_terms([words[0].split()])
        #print('Terms found: {}'.format([' '.join(t)for t in terms]))
    #print('----------------------------------------------------------')
    return [' '.join(t)for t in terms]

def test_koehler_data_experiments(args, type_name='db'):
    file_test = open_koehler_data(args.input_dir)
    print('Starting test..')
    #data = [' iron deficiency due to enemia', 'fluid injected with enemia']#validation_data.misspelled[1000000:1000102]
    if args.model_type ==  'noisy_model':
        corrector = ns.Sentence_Corrector(args.dict_root, args.output_dir, host=args.db_host, user=args.db_user, passwd=args.db_passwd)
        #max_len = 20
        type_rank = 'score'
    
    elif args.model_type ==  'bert_model':
        # root_pretrained, checkpoint_dir, dict_root, output_dir, using_mask=False, top_k = 3, only_non_word_errors=False):
        corrector = sc.SpellChecker(args.root_pretrained, args.checkpoint_dir, args.dict_root, args.output_dir, top_k=1)
        #max_len = 40
        type_rank = 'prob'
        type_name = ''

    for sentences in file_test:
        df_dummy = sentences[0].copy()
        fname = sentences[3]
        name_file = sentences[1]
        type_term = type_file(name_file)
        print("{} {}".format(fname,name_file))
        columns_name = ['TERM', 'CMT']
        col_name = [c for c in df_dummy.columns.values ]
        col_name += ['best_sentence_TERM']
        col_name += ['best_'+type_rank+'_TERM']
        col_name += ['time_TERM_'+type_name]
        col_name += ['terms_TERM_'+type_name]
        col_name += ['best_sentence_CMT']
        col_name += ['best_'+type_rank+'_CMT']
        col_name += ['time_CMT_'+type_name]
        col_name += ['terms_CMT_'+type_name]

        check_abbr = abbr.Abbreviation(dict_root=args.dict_root)  
        check_type =dt.DetectType(dict_root=args.dict_root,type_=type_term)

        df = pd.DataFrame(df_dummy,columns=col_name, dtype=object)
        del df_dummy
        for name in columns_name:
            sentences = df[name].values
            last_sentence = ''
            actual_sentence = ''
            last_correction = ''
            last_score = 0
            last_time = 0.0
            last_terms =[]
            for i,sentence in enumerate(tqdm(sentences)):
                #print("\rFold {}/{}.".format(i, len(sentences)), end='\r')
                #sys.stdout.flush()
                sentence = str(sentence)
                bestSentence = sentence
                bestScore = 0
                # if len(sentence.split()) > max_len + 1: 
                #     sentence = ' '.join([w for w in sentence.split()[:max_len]])
                if not sentence:
                    continue
                if i != 0:
                    last_sentence = sentences[i-1]
                    actual_sentence = sentence
                    if last_sentence == actual_sentence:
                        df.loc[df.index[i], 'best_sentence_'+name] = last_correction
                        df.loc[df.index[i], 'best_'+type_rank+'_'+name] = last_score
                        df.loc[df.index[i], 'time_'+name+'_'+type_name] = last_time
                        df.loc[df.index[i], 'terms_'+name+'_'+type_name] = last_terms
                        continue
                #sentence = ' '.join([w for w in sentence.split() if len(w) >2])
                sentence = check_abbr.change_abbreviation(sentence)
                bestSentence, bestScore,bestTime = corrector.return_best_sentence(sentence)
                if args.model_type ==  'bert_model':
                    if not isinstance(bestSentence,str):
                        bestSentence, bestScore,bestTime = bestSentence[0], bestScore[0],bestTime[0]
                    #bestSentence, bestScore,bestTime = bestSentence[0], bestScore[0],bestTime[0]
                df.loc[df.index[i], 'best_sentence_'+name] = bestSentence
                df.loc[df.index[i], 'best_'+type_rank+'_'+name] = bestScore
                df.loc[df.index[i], 'time_'+name+'_'+type_name] = bestTime
                terms = check_terms(check_type,bestSentence)
                df.loc[df.index[i], 'terms_'+name+'_'+type_name] = terms

                last_sentence = sentence
                last_correction = bestSentence
                last_score = bestScore
                last_time = bestTime
                last_terms=terms
                rewrite_file(df,args.output_dir,args.model_type,name_file,fname,i)

        rewrite_file(df,args.output_dir,args.model_type,name_file,fname, len(sentences))
        del df
        folder=re.findall(r'vitessa[\-*[a-z]*\.*[a-z]*]*',fname)[0]
        patient = sentences[2]
        print('The results have been stored at {}{}{}/{}'.format(args.output_dir,args.model_type,folder,patient))

def main():
    parser.add_argument("--input_dir",
                      default='../data/test/',
                      type=str,
                      required=False,
                      help="The input data directory where the folders and files as csv file are located.")
  
    parser.add_argument("--model_type",
                      default="bert_model",
                      type=str,
                      required=True,
                      help="Choose between noisy_model or bert_model, please.")

    parser.add_argument("--checkpoint_dir",
                      default="./bert_model/save_ckpt/",
                      type=str,
                      required=False,
                      help="The directory to save the checkpoint or where the last checkpoint was saved.")
    
    parser.add_argument("--root_pretrained",
                      default='./pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/',
                      type=str,
                      required=False,
                      help="The directory of the pre-trained clinicalBERT")

    parser.add_argument("--dict_root",
                      default='../dictionaries/',
                      type=str,
                      required=False,
                      help="The directory of the dictionary for the list of candidates")
    
    parser.add_argument("--output_dir",
                      default='../output/results/',
                      type=str,
                      required=False,
                      help="The directory where the results will be saved and/or updated.")
    
    parser.add_argument("--db_host",
                      type=str,
                      required=False,
                      help="Host database for the Noisy Channel Model.")
    parser.add_argument("--db_user",
                      type=str,
                      required=False,
                      help="The user name for getting access to the database for the Noisy Channel Model\
                            where the bigrams are located.")
    parser.add_argument("--db_passwd",
                      type=str,
                      required=False,
                      help="The password for getting access to the database for the Noisy Channel Model\
                            where the bigrams are located.")

    args = parser.parse_args()

    if args.model_type ==  'noisy_model':
        print('Noisy Channel Model has been chosen..')
    elif args.model_type == 'bert_model':
        print('Bert Model has been chosen..')
    else:
        print('Error, no {} has been found!'.format(args.model_type))
        print("In model_type, choose between noisy_model or bert_model, please.")
        exit()

    test_koehler_data_experiments(args)


if __name__ == "__main__":
    main()    