import pandas as pd 
import numpy as np
import re

class DetectType:
    def __init__(self, dict_root='../dictionaries/', type_='meddra'):
        #self.idx_cands
        self.dic_meddra = pd.read_csv(dict_root+'meddra_terms.csv')
        self.dic_meddra = self.dic_meddra.medical_terms.str.lower()
        self.dic_who = pd.read_csv(dict_root+'who_dic.csv')
        self.dic_who = self.dic_who.drugs_name.str.lower()
        self.type=type_
  
    def __check_terms__(self,sentence, words): 
        res = [all([k in s for k in words]) for s in sentence] 
        return [sentence[i] for i in range(0, len(res)) if res[i]] 

    def recursive_search(self,part,start,end):
        #d_part = part[start:start+2]
        d_part = part[start:end]
        idx = start
        idx_e = end
        len_terms = 0
        #print(d_part, idx_e)
        #print(len(check2(dic_meddra.values, d_part)))
        if self.type =='meddra':
            len_terms=len(self.__check_terms__(self.dic_meddra.values, d_part))
        elif self.type=='who':
            len_terms=len(self.__check_terms__(self.dic_who.values, d_part))
        return len_terms, idx, idx_e        

    def search_terms(self,word_l):
        terms = []
        for i,part in enumerate(word_l):
            #print('PART',part , len(part))
            found_smt = []
            idx_f = 0
            start = 0
            i_start = 0
            dummy_end = 0
            end = len(part)
            if len(part) >=2:
                while i_start < end: #
                    #print('Inside WHILE')
                    dummy_end += 1
                    candidates, idx, idx_e = self.recursive_search(part,start, dummy_end)
                    #print(part[start:dummy_end], candidates, idx, idx_e)
                    if candidates != 0:
                        #print(part[start:dummy_end])
                        start = idx
                        idx_f = idx
                        found_smt.append((part[start:dummy_end], idx_f))
                        #print(found_smt)
                    else:
                        start +=1
                    i_start +=1
            elif len(part) == 1 and end == 1:
                #print('Inside SECOND if')
                candidates, idx, idx_e = self.recursive_search(part,0, 0)
                if candidates != 0:
                    found_smt.append((part, 0))
            #print(len(found_smt))
            if found_smt:
                if not found_smt[0][0][0] in found_smt[-1][0]:
                    terms.append(found_smt[0][0])
                terms.append(found_smt[-1][0])
        #exit()
        return terms  

