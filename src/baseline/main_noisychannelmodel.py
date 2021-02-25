import argparse
import logging
import sys
import noisy_channel_model as ns
import os
import pandas as pd
import re

from tqdm import tqdm
from utils_ncm import clean_word, rewrite_file
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
  
    parser.add_argument("--dict_root",
                      default='../../dictionaries/',
                      type=str,
                      required=False,
                      help="The directory of the dictionary for the candidates")
    
    parser.add_argument("--output_dir",
                      default='../../output/results/baseline/',
                      type=str,
                      required=False,
                      help="The directory where the results will be saved and/or updated.")
    
    parser.add_argument("--output_dir_train",
                      default='../../output/train/',
                      type=str,
                      required=False,
                      help="The directory where the data after training the noisy channel model is saved.")
    
    parser.add_argument("--do_train",
                        default=False,
                        type=bool,
                        required=False,
                        help="For training")
    
    parser.add_argument("--db_host",
                      default='localhost',
                      type=str,
                      required=False,
                      help="Host database for the Noisy Channel Model.")

    parser.add_argument("--db_user",
                      default='arlette',
                      type=str,
                      required=False,
                      help="The user name for getting access to the database for the Noisy Channel Model\
                            where the bigrams are located.")

    parser.add_argument("--db_passwd",
                      default="Matthias16",
                      type=str,
                      required=False,
                      help="The password for getting access to the database for the Noisy Channel Model\
                            where the bigrams are located.")

    parser.add_argument("--max_num_candidates",
                      default=5,
                      type=int,
                      help="Stablish the number of candidates to be found for each misspellings.")

    args = parser.parse_args()
    noisy_model = ns.Noisy_Channel_Model(args.dict_root,output_dir=args.output_dir,output_dir_train=args.output_dir_train,
                                         train=args.do_train, host=args.db_host, user=args.db_user, passwd=args.db_passwd)
    start = 0
    for type_ in ['test']:#'train', 'val']:
        fname=args.data_dir+'misspelled_word_non_'+type_+'.csv' # misspelled_word_test #test_corpus
        data_experiments(noisy_model,fname, type_,start,args.output_dir)


def data_experiments(noisy_model, fname, type_, start, output_dir):
    print('Starting '+type_+' experiment..')
    data = pd.read_csv(fname) #open the file
    data = data.dropna().reset_index(drop=True) 
    df = pd.DataFrame(columns=['input_sentence', 'best_sentence','best_score','time', 'label'])
    errors = data.errors.values[start:]
    errors = ['diastolic turbulence of left carofid artery']#, 'fluid injected due to enemias', 'iron deficiency due to enemia']
    for i,sentence in enumerate(tqdm(errors), start=start):
        sentence = ' '.join([clean_word(w) for w in sentence.split()]) 
        print(sentence)
        bestSentence, bestScore, bestTime = noisy_model.return_best_sentence(sentence)
        print(bestSentence)
        # if not bestSentence:
        #     bestSentence = sentence
        # df = df.append({'input_sentence':sentence, 'best_sentence':bestSentence, 'best_score':bestScore, 'time':bestTime, 'label':data.original.values[i]}, ignore_index = True)
        # if (i % 100 == 0):
        #     #print('Loop {}, the results have been stored at ./output/results/baseline/noisychannel_out_{}.csv'.format(i,type_))
        #     #print('---------------------------------------------------------------------------------------------------------------------')
        #     #rewrite_file(df, type_, output_dir=output_dir)
        #     df = pd.DataFrame(columns=['input_sentence', 'best_sentence','best_score','time', 'label'])
        # print(df['best_sentence'])    
    print('Almost done')
    #rewrite_file(df, type_,output_dir=output_dir)


if __name__ == "__main__":
    main()