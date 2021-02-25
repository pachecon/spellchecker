import difflib
import enchant
import numpy as np
import pandas as pd
import sys

from math import *
from sklearn.utils import shuffle
from metaphone import doublemetaphone
from utils_mg import (read_file, rewrite_file,create_misspellings_file,
                        prepare_indexes_random)

class RealWordTypos:
    """
    Replace word(s) of the sentence with correct words from the lexico; based on the edit distance of 2 
    and their phonetic relation between these suggestions and the original word(s)
    
    Methods
    --------
    word_errors
    __prepare_indexes_random__
    __candidate_sentence__
    __prepare_word_errors_line__
    """
    def __init__(self,type_v, name_file, dict_root='../../dictionaries/', output_root='../../data/misspelled_corpora/'):
        self.dictionary = enchant.DictWithPWL("en_US", dict_root+'multilabel_dic_unique_order.csv')
        self.fname = name_file
        self.type_v = type_v
        self.output_root = output_root
        self.output_fname = 'non_'+self.type_v

    def __create_phonetics__(self,word,misspelling_list):
        word_phon = doublemetaphone(word)[0]            
        new_error_list = []
        for error in misspelling_list:
            error_phon = doublemetaphone(error)[0]
            if(len(error_phon) !=0 and (error_phon == word_phon)):
                new_error_list.append(error)
        return new_error_list,error_phon
    
    def __sort_dict__(self,unsorted_dict):
        return sorted(unsorted_dict.items(), key=lambda t: t[1], reverse=True)

    def __random_word_errors__(self,word):
        suggests,max = {},0
        a = set(self.dictionary.suggest(word))
        #print('WOrd suggested: {}'.format(a))
        a,_ = self.__create_phonetics__(word,a)
        for b in a:
            tmp = difflib.SequenceMatcher(None, word, b).ratio()
            if tmp == 1.0:
                continue
            suggests[b] = tmp
            if tmp > max:
                max = tmp
        suggests = self.__sort_dict__(suggests)
        if len(suggests)>3:
            suggests = suggests[:3]
        elif len(suggests) < 3:
            for _ in range (3-len(suggests)):
                suggests.append((' ',0.0))
        #print('Word: {}, Suggestions:{}'.format(word, suggests))
        return suggests, len(suggests)
    
    def __candidate_sentence__(self,list_line,n):
        """
        Takes one sentence, and return all the possible sentences, and also return a dictionary of word : suggested number of words
        """
        words_count = {}
        df_cna = pd.DataFrame(columns=n)
        df_probs = pd.DataFrame(columns=n)
        for i,word in enumerate(list_line):
            try:
                candidates_tuple, num_candidates = self.__random_word_errors__(word)
                df_cna[n[i]] = [c[0] for c in candidates_tuple] 
                df_probs[n[i]] = [np.round(c[1],3) for c in candidates_tuple] 
                words_count[n[i]] = num_candidates
            except:
                continue
        return df_cna, df_probs, words_count#candidate_sentences, probs, words_count

    def __prepare_word_errors_line__(self,line,words_changed,n,probs,words_count):
        new_sent_error = []
        probabilities = []
        dummy = list(line)
        #print('Dummy ', dummy)
        total_can = len(words_changed)
        
        for can_num in range(total_can):
            try:
                for y, idx in enumerate(n):
                    dummy[idx] = words_changed[idx][can_num]
                new_sent_error.append(' '.join(list(dummy)))
                probabilities.append(list(probs.iloc[can_num, :].values))
            except:
                continue
        return new_sent_error, probabilities

    def word_errors(self):
        data = read_file(self.fname)
        self.output_fname = 'error_'+self.type_v
        dic_new = pd.DataFrame(columns=['original', 'errors', 'probabilities']) #, 'phonetic'
        print("Create random misspellings (word errors) ...")
        for i,line in enumerate(data[:1010]):#892856:
            print("\rFold {}/{}.".format(i+1, len(data[892856:])), end='\r')
            sys.stdout.flush()
            original = line
            line = line.split()
            new_errors = []
            probs = []
            words_changed = []
            probs = []
            if len(line) >1:
                n = prepare_indexes_random(line)
                list_line = [line[idx] for idx in n]
                words_changed, probs,words_count = self.__candidate_sentence__(list_line,n)
                new_errors, probs = self.__prepare_word_errors_line__(line,words_changed,n,probs,words_count)
                original = [original] * len(new_errors)
            else:
                continue
            #print(original, new_errors, probs)
            dic_new = create_misspellings_file(dic_new, original, new_errors,probs=probs)
            #print(dic_new)
            if(i%1000 == 0):
                rewrite_file(dic_new, self.output_root, self.output_fname)
                dic_new = pd.DataFrame(columns=['original', 'errors','probabilities'])      #, 'phonetic'
        rewrite_file(dic_new, self.output_root, self.output_fname)
        print('Misspelled words have been saved at "../data/misspelled_corpora/misspelled_word_'+self.output_fname+'.csv"')
            