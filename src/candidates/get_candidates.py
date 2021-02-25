import difflib
import enchant
import itertools
import numpy as np 
import pandas as pd

class Candidates:
    """
    Attributes
    ----------
    d: enchant.DictWithPWL
        It is the dictionary which is been used for the elaboration of the candidates given a key word. 
        It is used to combine a language dictionary and a custom dictionary also known as Personal Word List(PSL). 
        This dictionary recieves the path of the complete dictionaries from WHO-Drug and MEDDRA-llt
    limit_candidates: int
        The user can specify the limits of the candidates, if desired.
        Default is 0 because it is used as a flag. When a value is given, then the list of suggested candidates
        are limited to the desired value
    
    Methods
    -------
    check_in_dict(word):
        check whether a word is part of the dictionary "d"
    __suggest_candidates__(word):
        return list of suggested words obtained from the PyEnchant library
    __calculate_edit_distance__(word1, word2):
        return edit distance between two strings
    __sort_dict__(unsorted_list):
        sort the list of candidates from higher similarity to lower
    candidate_word(word):
        get suggestions of candidates, calculate their edit distance, limit number of suggestions (if desired),
        order the list of candidates based on their similarity with respect to the word, and check whether the word is part of the dictionary "d"
    __combine_candidates__(sentence, candidates):
        replace each candidate once for real-word spelling errors anf return list of new sentences
    __combine_non_error_candidates__(sentence, candidates, in_dict_list):
        This function is for replacing only the non-word spelling errors with the candidates
    __get_index_error__(lst, item):
        Get the position (index) of those words whose value is False i.e. they are misspellings
    get_candidate_sentence(sentence):
        Get a list of new sentences based on the candidates 
    """

    def __init__(self, dict_root, limit_candidates = 0):
        """
        Parameters
        ----------
        dict_root : str
            The directory where the custom dictionary is located (ours). 
        limit_candidates : int (Optional)
            The user can specify the limits of the candidates, if desired.
            Default is 0 because it is used as a flag. When a value is given, then the list of suggested candidates
            are limited to the desired value
        """
        self.d = enchant.DictWithPWL("en_US", dict_root+'multilabel_dic_unique_order.csv')
        stop_=dict_root+'stop_words.txt'
        stop_ = pd.read_csv(stop_, header=None)
        self.stop_words = stop_[0].values
        self.limit_candidates = limit_candidates

    def check_in_dict(self, word):
        """
        Check whether a word is part of the Dict object "d"
        Parameters
        ----------
        word: str
            the word that is searched in the Dict object "d" 
        
        Return
        ------
            boolean
        """
        return self.d.check(word)
    
    def __suggest_candidates__(self,word):
        """
        The PyEnchant library suggests the candidates for the word
        Parameters
        ----------
        word: str
            the word that is used for returning its candidates 
        
        Return
        ------
        list
            list of candidates found for the word
        """
        if word in self.stop_words:
            return [word]
        else:
            return self.d.suggest(word)
    
    def __calculate_edit_distance__(self, word1, word2):
        """
        edit distance between two strings is defined as the minimum number of characters needed to 
        insert, delete or replace in a given string word1 to transform it to another string word2.
        return the edit distance
        Parameters
        ----------
        word1: str
            the first word 
        word2: str
            the second word

        Return
        ------
        int
            return the edit distance
        """
        return enchant.utils.levenshtein(word1, word2)

    def __sort_dict__(self, unsorted_list):
        """
        Order the list of candidates from higher similarity  between the word and each candidate to
        the lower similarity
        Parameters
        ----------
        list
            unsorted list of candidates with its similarity value with respect to the word
        Return
        -------
        list
            sorted list of candidates from higher similarity to lower
        """
        return sorted(unsorted_list.items(), key=lambda t: t[1], reverse=True)

    def candidate_word(self, word):
        """
        Generate similar word(s) for a given word based on their edit distance, order the candidates based
        on their similarity with difflib library  and verify whether the word is found in the dictionary "d". 
        Limit number of suggestions (if desired)
        Parameters
        ----------
        word: str
            the word that is used for returning its candidates 
        Return
        -------
        tuple (list, int, bool)
            (list of candidates, length of the list, whether the word is part of "d" )
        """
        suggests,max = {},0
        try:
            possible_candidates =set(self.__suggest_candidates__(word)) 
        except:
            possible_candidates = set('no')
        in_dic = 'False'
        if self.d.check(word):
            possible_candidates.add(word)
            in_dic = 'True'
        #print('WOrd suggested: {}'.format(possible_candidates))
        
        for b in possible_candidates:
            edit_distance = self.__calculate_edit_distance__(word,b)
            if edit_distance <= 2:#edit_distance != 0 and 
                tmp = difflib.SequenceMatcher(None, word, b).ratio()
                suggests[b] = tmp
                if tmp > max:
                    max = tmp
        #print('BEFORE Suggestions length: ',len(suggests))
        suggests = self.__sort_dict__(suggests)
        if self.limit_candidates != 0 and (len(suggests) >self.limit_candidates):
            suggests = suggests[:self.limit_candidates]
        # print('AFTER Suggestions length: ',len(suggests))
        if word == 'enemia' or word == 'enemias':
            print('SUGGESTIONS: ', suggests)
        #labels = self.get_labels(suggests)
        return suggests, len(suggests),in_dic#, labels
        
    def __combine_candidates__(self, sentence, candidates):
        """
        This function is for replacing all words of the input sentence for each of its candidates, 
        this is needed when all words are found in the dictionary (real-word spelling errors).
        we are considered that only an error occurs per sentence. For an example please go to the Thesis in Appendix Candidates
        Parameters
        -----------
        sentence: list
            the original input tokenized sentence i.e. the sentence is already separated by tokens. 
            This is used as pivot for the replacement of a word with its candidates 
        candidates: list
            list of candidates for each word of the input sentence
        Return
        -------
        list
            list of new sentences where only one candidate appear in each new sentence i.e. we are considered that only
            an error occurs per sentence.
        """
        new_sentences = []
        dummy = ' '.join(sentence)
        for i,word in enumerate(sentence):
            for c in candidates[i]:
                if len(c.split()) > 1:
                    c =''.join(c.split()) 
                dummy_replacement = dummy.replace(word, c)
                new_sentences.append(dummy_replacement.split())
        return new_sentences
    
    def __get_index_error__(self, lst, item):
        """
        We want the position (index) of those words whose value is False. This means, these words were not found in the dictionary. Therefore,
        they are considered as non-word spelling errors.
        Parameters
        -----------
        lst: list
            list of False and True. True means that the word was found in the dictionary and False means that the word was not found in the dictionary
        item: str
            we are only interesting in the position of those words which were not found in the dictionary (False)
        Return
        -------
        list
            all the indexes where False is found i.e. the position of the misspellings. False means that the word was not found in the dic "d"
        """
        return [i for i, x in enumerate(lst) if x == item]

    def __combine_non_error_candidates__(self, sentence, candidates, in_dict_list):
        """
        This function is for replacing only the non-word spelling errors; which are the misspellings not found in the dic "d", with the candidates.
        we are considered that only an error occurs per sentence. For an example please go to the Thesis in Appendix Candidates
        Parameters
        -----------
        sentence: list
            the original input tokenized sentence i.e. the sentence is already separated by tokens. 
            This is used as pivot for the replacement of a word with its candidates 
        candidates: list
            list of candidates for each misspelling of the input sentence
        in_dict_list: list
            list of indexes where the misspellings where located in the input sentence
        Return
        -------
        list
            list of new sentences where only one candidate appear in each new sentence i.e. we are considered that only
            an error occurs per sentence.
        """
        new_sentences = []
        error_idx = self.__get_index_error__(in_dict_list, 'False')
        candidates_error = [c for i,c in enumerate(candidates) if i in error_idx]
        new_cand_error = []
        for can_err in candidates_error:
            dummy = []
            for c in can_err:
                if len(c.split()) > 1:
                    c = ''.join(c.split())
                dummy.append(c)
            new_cand_error.append(dummy)
        candidates_error = new_cand_error
        candidates_error = list(itertools.product(*candidates_error))
        dummy = ' '.join(sentence)
        for c_combination in candidates_error:
            for j,idx in enumerate(error_idx):
                dummy = dummy.replace(sentence[idx], c_combination[j])
            new_sentences.append(dummy.split())
            dummy = ' '.join(sentence)
        return new_sentences
    
    def get_candidate_sentence(self, sentence):
        """
        Takes an original input tokenized sentence, get the candidates, check whether the words are foun in the 
        dictionary "d", if the sentence has non-word spelling errors, then, get a list of new sentences based on 
        the candidates  and return all the possible sentences, 
        and also return a dictionary of word : suggested number of words
        Parameters
        ----------
        sentence: list 
            the original input tokenized sentence i.e. the sentence is already separated by tokens
        Return
        --------
        list 
            list of new sentences with each candidate per sentence
        dict
            number of candidates which each misspelling have. This is only used for the Noisy Channel Model
        -------
        """
        candidate_sentences = []
        words_count = {}
        in_dict_list = []
        for i,word in enumerate(sentence):
            #if not word:
            #    continue
            candidates_tuple, num_suggests, in_dict = self.candidate_word(word)
            candidate_sentences.append([c[0] for c in candidates_tuple])
            words_count[word] = num_suggests
            in_dict_list.append(in_dict)
        #print(in_dict_list)
        #print(candidate_sentences)
        #print('Candidate sentences len: ',len(candidate_sentences))
        if not ('False' in in_dict_list):
            #print('All words are found in dict')
            candidate_sentences = self.__combine_candidates__(sentence, candidate_sentences)
        else:
            candidate_sentences = self.__combine_non_error_candidates__(sentence, candidate_sentences,in_dict_list)
        #print(candidate_sentences)
        #print(len(candidate_sentences))
        #print('Len of candidates: ',len(candidate_sentences))
        return candidate_sentences, words_count

if __name__ == "__main__":
    can = Candidates('../../dictionaries/')
    print(can.candidate_word('metatadsal'))