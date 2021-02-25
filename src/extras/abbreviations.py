import pandas as pd 

class Abbreviation:
    def __init__(self,dict_root='../dictionaries/'):
        self.abbreviation_list = pd.read_csv(dict_root+'abbreviation_list.csv')
        self.abbreviation_list.abbr = self.abbreviation_list.abbr.str.lower()
    
    def change_abbreviation(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.split()
        dummy = ' '.join([w for w in sentence])
        #dummy = [dummy.replace(sub, abbreviation_list['meaning'].loc[sub == abbreviation_list['abbr']].values[0]) 
        #            for sub in sentence if sub in abbreviation_list['abbr'].values] 
        for i,word in enumerate(sentence):
            if word in self.abbreviation_list['abbr'].values:
                meaning = self.abbreviation_list['meaning'].loc[word == self.abbreviation_list['abbr']].values[0]
                if(not isinstance(meaning,str) and len(meaning)>1):
                    meaning = meaning[0]
                dummy=dummy.replace(word, meaning)
        #print('Sentence without abbreviations: {}'.format(dummy))
        return dummy #' '.join([w for w in sentence])