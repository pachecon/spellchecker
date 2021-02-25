import math
import numpy as np
import os
import pandas as pd 
import re
from os import walk

def rewrite_file(df,output_dir, type_model, type_exp,fname,i):
    """
    Rewrite the KOEHLER files with the modified sentences
    """
    folder = re.findall(r'vitessa[\-*[a-z]*\.*[a-z]*]*',fname)[0]
    patient = re.findall(r'\b(P[0-9]+)\b',fname)[0]
    if not (os.path.exists(output_dir+type_model+'/'+folder+'/'+patient+'/')):
        os.makedirs(output_dir+type_model+'/'+folder+'/'+patient+'/')
    df.to_csv(output_dir+type_model+'/'+folder+'/'+patient+'/'+type_model+'_out_'+type_exp+'.csv', index = False)

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)              
    return allFiles        

def prepare_list_files(root):
    allfiles = []
    for (dirpath, dirnames, filenames) in walk(root):
        for name in dirnames:
            #print (os.path.join(dirpath, name))
            allfiles.append(getListOfFiles(os.path.join(dirpath, name)))
        break
    return allfiles
    
def clean_sentence(text):
    """
    pattern = r"(\s+|(?:[A-Z']\.?)+)"
    dummy_sentence = [t for t in re.split(pattern, line, flags=re.I) if t and not t.isspace()]   
    """
    if not isinstance(text, str):
        if text is None or math.isnan(text):
            return ''
    text = text.lower() 
    text = re.sub(r'\"+', '', text)
    text = text.strip()
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'[{}@_*>\\#%+=\[\]\-]',' ', text) #()
    text = re.sub(r'a0',r'', text)
    text = re.sub(r'\'92t',r'\'t', text)
    text = re.sub(r'\'92s',r'\'s', text)
    text = re.sub(r'\'92m',r'\'m', text)
    text = re.sub(r'\'92ll',r'\'ll', text)
    #text = re.sub(r'\'91',r'', text)
    #text = re.sub(r'\'92',r'', text)
    #text = re.sub(r'\'93',r'', text)
    #text = re.sub(r'\'94',r'', text)
    #text = re.sub(r'\.',r'', text)
    text = re.sub(r'\,',r' ', text)
    text = re.sub(r'\!',r' ', text)
    text = re.sub(r'\?',r' ', text)
    text = re.sub(r' +',r' ', text)
    text = re.sub(r'\/',r' ', text)
    text = re.sub(r'[\b\ufeff\b]',r' ',text)
    #text = re.sub(r'[0-9]+',r'',text)
    return text

def clean_test_file(fname):
    df = pd.read_csv(fname)
    #df = df.dropna()
    df = df.reset_index(drop=True)
    type_name = 'no'

    if re.findall(r'(meddra_)',fname):
        df.TERM = pd.Series([clean_sentence(s) for s in df.TERM ])
        df.CMT = pd.Series([clean_sentence(s) for s in df.CMT ])
        source = df.SOURCE[0]
        #df = df.sort_values(by = ['TERM'])
        #print(df.head())
        type_name = 'meddra_'+source #re.findall(r'\b(meddra_[a-z]*)\b',fname)[0]
    elif re.findall(r'(who_[a-z]*)',fname):
        if re.findall(r'(who_visits)',fname):
            df = df.fillna(' ')
            term = df.TERM1.map(str) + ' ' +df.TERM2.map(str)+ ' ' + df.TERM3.map(str)
            df['TERM'] = term
        df.TERM = pd.Series([clean_sentence(s) for s in df.TERM ])
        df.CMT = pd.Series([clean_sentence(s) for s in df.CMT ])
        source = df.SOURCE[0]
        type_name = 'who_'+source
    return df, type_name

def open_koehler_data(root):
    paths = prepare_list_files(root)
    for path in paths:
        for fname in path:
            if re.findall(r'\b(\.csv)\b',fname):
                #print('It is a csv file')
                patient = re.findall(r'\b(P[0-9]+)\b',fname)[0]
                if re.findall(r'\ball_queries\b',fname) or re.findall(r'\ball_querynodes\b',fname):
                    continue
                validation_data, type_name = clean_test_file(fname)
                yield (validation_data, type_name,patient,fname)

def type_file(name_file):
    if re.findall(r'[^\bmeddra\b]+',name_file):
        return 'meddra'
    elif re.findall(r'[^\bwho\b]+',name_file):
        return 'who'