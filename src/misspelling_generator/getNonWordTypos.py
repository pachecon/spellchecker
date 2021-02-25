import difflib
import enchant
import random
import numpy as np
import pandas as pd
import string
import sys

import getTyposKB as kb 

from math import *
from sklearn.utils import shuffle
from random import sample, choice 
from utils_mg import (read_file, rewrite_file,create_misspellings_file,
                        prepare_indexes_random)

class NonWordTypos:
    """
    Creat non-word misspellings which are words not found in a dictionary
    Randomly selects a word from the sentence and modify any character based on:
        1)Edit distance 
            -Insertion
            -Deletion
            -Substitution
            -Transposition
        2)Keyboard algorithm
            -a character (key) is change with any of the characters which are found closer to the key based on
            the german keyboard
    Attributes
    ----------
    dictionary: enchant.DictWithPWL object
        initialize the lexico with our list of words and the english dictionary from enchant library.
    fname:str
        the name of the file where the correct sentences are found.
    type_v:str
        the type of the file which can be train, val or test.
    output_root:
        directory of the csv files with the sentences with misspellings and the original correct sentences are saved.
    output_fname:str
        non+type: for non words misspellings and error+type: for real word misspellings
    typosKB:TyposKB object
        create an object for the keyboard algorithm
    Methods
    --------
    misspelling_random_choice(word):
        Randomly choose a character from the word change it with any character from string.ascii_letters
        
    """
    def __init__(self,type_v, name_file, dict_root='../../dictionaries/', output_root='../../data/misspelled_corpora/'):
        """
        Parameters
        -----------
        type_v:str
            the type of the file which can be train, val or test.
        name_file:str
            the name of the file where the correct sentences are found.
        dict_root:str
            directory of our lexico
        output_root:
            directory of the csv files with the sentences with misspellings and the original correct sentences are saved.
        """
        self.dictionary = enchant.DictWithPWL("en_US", dict_root+'multilabel_dic_unique_order.csv')
        self.fname = name_file
        self.type_v = type_v
        self.output_root = output_root
        self.output_fname = 'non_'+self.type_v
        self.typosKB = kb.TyposKB()

    def misspelling_random_choice(self,word):
        """
        Randomly choose a character from the word change it with any character from string.ascii_letters
        Parameters
        ----------
        word:str
            the correct word without any modifications
        Return
        -------
        str:
            the modified word
        """
        ix = choice(range(len(word)))
        new_word = ''.join([word[w] if w != ix else random.choice(string.ascii_letters.lower()) for w in range(len(word))])
        return new_word
    
    
    def misspelling_with_noise(self,word, threshold= 0.3):
        new_word = ""
        random = np.random.uniform(0,1,1)
        if random < threshold:
            #print('First option')
            # select random positions in token
            p1 = choice(range(len(word)))#sample(range(len(word)), 2)
            p2 = choice(range(len(word)))
            # swap the positions
            if p1 == p2:
                self.misspelling_with_noise(word)
            else:
                l = list(word)
                #for first, second in zip(p1,p2): #zip(positions[::2], positions[1::2]):
                #    l[first],l[second]= l[second],l[first]
                l[p1],l[p2]= l[p2],l[p1]
                new_word = ''.join(l)
        elif random < 0.8 and random > 0.3:
            #print('Second option')
            idx = choice(range(len(word)))
            new_word = word.replace(word[idx], '',1)
        else:
            #print('Third option')
            new_word = self.misspelling_random_choice(word)
        return new_word

    def create_random_misspellings(self,word):
        word = str(word).rstrip()
        misspellings_list= self.typosKB.getTypos_based_kb(word)
        new_word = self.misspelling_with_noise(word)
        if new_word not in misspellings_list:
            misspellings_list.append(new_word)
        return misspellings_list

    def change_word(self,list_line):
        words_changed = []
        for word in list_line:
            if not len(word) > 3 or word.isupper():
                #print('Small word: ',word)
                words_changed.append(word)
                continue
            misspelling = self.create_random_misspellings(word)
            #misspelling,_ = create_phonetics(word,misspelling) #phonetic
            #print('Misspellings_list: ',misspell)
            if not misspelling:
                while not misspelling:
                    misspelling = self.create_random_misspellings(word)
                    if misspelling:
                        break
            n = random.randint(0,len(misspelling)-1)

            words_changed.append(misspelling[n])
            #phonetics.append(phonetic)
        return words_changed

    def prepare_errors_line(self,line,words_changed,n):
        for y, idx in enumerate(n):
            line[idx] = words_changed[y]
        #phonetic_sentence = ' '.join(phonetics)
        #print(line, words_changed, n)
        line = ' '.join(line)
        return line

    def create_edit_misspellings(self, k=2):
        data = read_file(self.fname)
        dic_new = pd.DataFrame(columns=['original', 'errors']) #, 'phonetic'
        print("Create random misspellings (non-word errors)...")
        
        for i,line in enumerate(data[:2010]):#:892856
            try:
                print("\rFold {}/{}.".format(i+1, len(data[:892856])), end='\r')
                sys.stdout.flush()
                original = line
                line = line.split()

                if len(line) >1:
                    n = prepare_indexes_random(line,k)
                    list_line = [line[idx] for idx in n]
                    words_changed = self.change_word(list_line)
                    line = self.prepare_errors_line(line,words_changed,n)
                else:
                    line = self.change_word(line)
                dic_new = create_misspellings_file(dic_new, original, line)
                if(i%1000 == 0):
                    rewrite_file(dic_new, self.output_root, self.output_fname)
                    dic_new = pd.DataFrame(columns=['original', 'errors'])      #, 'phonetic'
            except:
                continue
        rewrite_file(dic_new, self.output_root, self.output_fname)
        print('Misspelled words have been saved at "../data/misspelled_corpora/misspelled_word_'+self.output_fname+'.csv"')