import sys
sys.path.insert(1, '../candidates/')

import collections
import connect_database as cdb
import difflib
import get_candidates
import math 
import numpy as np
import os
import pandas as pd
import re
import time

from collections import defaultdict
from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance
from utils_ncm import tokenize_file, save_total, save_ngrams, load_pickle, load_ngram, clean_word

class Noisy_Channel_Model:
    """
    This class works with the algorithm Noisy Channel Model
    Given a set of candidates, calculate a score by passing each candidate to the Noisy Channel and get the score of being
    modified as the observed misspell.
    The score also depends of the value of the Prior. 

    Attributes
    ----------
    get_candidates: Candidates
        prepare the Candidates class for searching the candidates for each misspelling
    OUT_PATH: str
        the directory for the output, where the results after correction the sentences are saved.
        Default = output_dir+'baseline/'
    OUT_PATH_TRAIN: str
        the path for the output where the bigrams and unigrams will be saved after TRAINING
    db: NgramsDB object
        (host,user,password) to connect to MySQL database where the bigrams are saved.
    sentences: list
        all sentences use for TRAINING, where the n-grams will be obtained
    unigramCounts: collections.defaultdict
        All unique words are saved as unigrams during TRAINING with its number of times the same word is repeated.
        If training is over, then they are loaded to be used.  
    bigramCounts: collections.defaultdict
        All pair od words are saved as bigrams during TRAINING with its number of times the same pair is repeated.
        If training is over, then they are loaded to be used.  
    total: int
        The total number of words during TRAINING.
        If training is over, then they are loaded to be used.  
    
    Methods
    -------
    train()
    channel_model(words_count, new_sentence, old_sentence)
    backoff_model(sentence) 
    return_best_sentence(old_sentence) 
    """
    def __init__(self,dict_root, output_dir='../output/results/', input_data_training='../../data/train/train_dataset.csv', 
                    output_dir_train='../output/train/', train=False, 
                    host="localhost", user="arlette", passwd="Matthias16"):
        """
        Parameters
        -----------
        dict_root: str
            the path where the dictionary is located.
        output_dir: str (Optional)
            Default '../output/results/'
            The path where the outputs will be saved 
        input_data_training:str (Optional)
            Default '../../data/train/train_dataset.csv'
            The path where the training dataset is located. From this data, the n-grams will be obtained and saved during TRAINING
        output_dir_train:str (Optional)
            Default '../output/train/'
            The path where the file used for training is located
        train:bool (Optional)
            Default False ; means the model has already been trained.
            Otherwise, the model will be trained for obtaining the unigrams and bigrams from the train dataset
        host:str (Optional)
            The host for MySQL database. Default = "localhost"
        user:str (Optional)
            The user of MySQL database.
        passwd:str (Optional)
            The password used to connect to MySQL atabase.
        """
        self.get_candidates = get_candidates.Candidates(dict_root)
        self.OUT_PATH = output_dir
        self.OUT_PATH_TRAIN = output_dir_train
        self.db = cdb.NgramsDB(host,user,passwd)#Matthias16 arlette
        if train:
            self.unigramCounts = collections.defaultdict(lambda: 1)
            self.bigramCounts = collections.defaultdict(lambda: 1)
            #self.trigramCounts = collections.defaultdict(lambda: 1)
            self.total = 0
            self.sentences = tokenize_file(input_data_training)
            self.train()
        else:
            self.total = load_pickle(self.OUT_PATH_TRAIN,'total')
            unigramCounts = load_ngram(self.OUT_PATH_TRAIN,'unigrams')
            self.unigramCounts = collections.defaultdict(lambda: 1)
            
            if(not isinstance(unigramCounts,collections.defaultdict)):
                for i,key in enumerate(unigramCounts.key):
                    self.unigramCounts[key] = unigramCounts.value[i] 
                del unigramCounts
        self.counting = self.unigramCounts
        

    def train(self):
        """
        If the user specifies to train the model. Then this function is called.
        The unigrams and bigrams are created from the training data set.
        Both n-grams will be saved as csv files.

        """
        print('Train the baseline model (noisy channel)')
        for i,sentence in enumerate(self.sentences):
            print("\rFold {}/{}.".format(i, len(self.sentences)), end='\r')
            sys.stdout.flush()
            sentence.insert(0, '<s>')
            sentence.append('</s>')
            for i in range(len(sentence) - 1):
                try:
                    token1 = sentence[i]
                    token2 = sentence[i + 1]
                    self.unigramCounts[token1] += 1
                    self.bigramCounts[token1+' '+token2] += 1
                    #self.trigramCounts[(token1, token2, token3)] += 1 
                    self.total += 1
                except :
                    continue
            self.total += 1
            self.unigramCounts[sentence[-1]] += 1
        save_ngrams(self.OUT_PATH_TRAIN,self.unigramCounts, 'unigrams')
        save_ngrams(self.OUT_PATH_TRAIN,self.bigramCounts, 'bigrams')
        save_total(self.OUT_PATH_TRAIN,self.total, 'total')

    def __get_ngrams_db__(self,bigram_key):
        """
        Search the query in the database 
        Parameter
        ---------
        bigram_key:str
            the query to be searched in the table of the database
        Return
        ------
        int
            the number of times that query was found. This number is obtained during Training
        """
        query = "SELECT counts_number FROM ngrams2 WHERE bigrams_key = '"+bigram_key+"'"
        return self.db.get_data_from_db(query)

    def channel_model(self, words_count, new_sentence, old_sentence) :
        """
        Take an old sentence and a new sentence, for each words in the new sentence, if it's same as the orginal sentence, assign 0.95 prob
        If it's not same as original sentence, give 0.05 / (count(similarword) - 1). Otherwise 0.
        When p is close to zero or one, the distribution is very skewed, and the distribution for p equal to p1 is the mirror 
        image of the distribution for p equal to (1–p1)
        Parameters
        ----------
        words_count: dict()
            The number of candidates that each word has    
        new_sentence: list
            The tokenized sentences with the candidates.
        old_sentence: list
            The original tokenized input sentence
        Return
        -------
        float
            The score 
        """
        score = 1
        print(new_sentence)
        for i in range(len(new_sentence)):
            try:
                if new_sentence[i] in words_count :
                    score *= 0.95
                
                #elif self.unigramCounts[new_sentence[i]] == 1:
                #    score *= 0.05
                else :
                    print(old_sentence[i], new_sentence[i])
                    #score *=(0.05/(damerau_levenshtein_distance(old_sentence[i], new_sentence[i])))
                    score *= (0.05*self.counting[new_sentence[i]] / (len(self.counting)))
                # elif words_count[old_sentence[i]] == 1:
                #     score *= 0.05
                # else :
                #     score *= (0.05 / (words_count[old_sentence[i]] - 1))
            except :
                score = 0
        
        if score != 0:
            print(score)
            return math.log(score)
        else:
            return 0

    def backoff_model(self, sentence):
        """
        Takes a list of strings as argument and returns the log-probability of the
        sentence using the stupid backoff language model.

        Parameters
        ----------
        sentence: list
            The tokenized sentence with candidates.
        Return
        ------
        float
            The score after using the Stupid Backoff algorithm
        """
        score = 0.0
        #clean_sentence
        for i in range(len(sentence) - 1):
            value = self.__get_ngrams_db__(sentence[i]+" "+sentence[i+1])
            if value is None:
                value = 0
            if not isinstance(value, (float,int)):
                value = value[0][0]
            if  value > 0:
                score += (math.log(value))#/math.log(self.unigramCounts[sentence[i]]))
                score -= math.log(self.unigramCounts[sentence[i]])
            else:
                score += ((math.log(self.unigramCounts[sentence[i + 1]] + 1) + math.log(0.4)))#/math.log(self.total*2))
                score -= math.log(self.total*2)#+ len(self.unigramCounts))
        return 0.98*score

    
    def return_best_sentence(self, old_sentence):
        """
        Generate all candiate sentences and calculate the score of each one 
        Return the one with highest score.
        Probability involves two parts: 1. noisy channel and 2. prior model
        noisy channel : p(c | w)
        prior model : use stupid backoff algorithm

        For the  backoff model, we need to prepare the sentence with initial and ending tokens
        which are represented with <s> and </s>
        
        Parameters
        ----------
        old_sentence:list
            input sentence which is already tokenized
        
        Return
        -------
        tuple (str, float, float)
            bestSentence, bestScore, final_time
        """
        bestScore = float('-inf')
        bestSentence = []
        old_sentence = old_sentence.lower()
        old_sentence =[clean_word(s) for s in old_sentence.split()]
        old_sentence = list(filter(None, old_sentence))
        sentences, word_count = self.get_candidates.get_candidate_sentence(old_sentence)
        print(sentences, word_count)
        #sentences.append(['diastolic turbulence of left carofid artery'])
        start = time.time()
        for i, new_sentence in enumerate(sentences):
            #print("\rFold sentences {}/{}.".format(i, len(sentences)), end='\r')
            #sys.stdout.flush()
            new_sentence = list(new_sentence)
            score = self.channel_model(word_count, new_sentence, old_sentence) 
            print(score)
            new_sentence.insert(0, '<s>')
            new_sentence.append('</s>')
            score_back = self.backoff_model(new_sentence) 
            score += score_back
            print(score_back)
            print(score)
            print('------------------------------------------------------------')
            if score >= bestScore: #Keep the maximum score
                bestScore = score
                bestSentence = new_sentence
        end = time.time()
        bestSentence = ' '.join(bestSentence[1:-1])
        self.db.close_db()
        final_time = end-start
        return bestSentence, bestScore, final_time

if __name__ == "__main__":
    sentences =['there is twaring of the membranous portion of the ligament']
    corrector = Noisy_Channel_Model('../../dictionaries/', output_dir_train='../../output/train/',passwd='arlette')
    for s in sentences:
        print(s)
        start = time.time()
        print(corrector.return_best_sentence(s))
        end = time.time()
        print(end-start)
        
    