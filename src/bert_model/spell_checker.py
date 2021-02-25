import sys
sys.path.insert(1, '../candidates/')

import difflib
import enchant
import itertools
import numpy as np 
import os
import pandas as pd
import torch
import time
import sys

import mask_lang_model as mlm
import get_candidates as cc

from scipy.spatial import distance
from tqdm import tqdm
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSequenceClassification
from utils_bert import (prepare_dataloader,create_attention_mask,prepare_inputs_dataloader, 
                    load_model_detection, clean_sentence, append_results)

class SpellChecker:
    """
    Attributes
    ----------
    out_dir: str
        the directory where the results will be saved.
    tokenizer: trasnformers.AutoTokenizer
    c_model:transformers.modeling_bert.BertForSequenceClassification
    using_mask: bool
    ml_model = mlm.MaskLanguageModel(root_pretrained, dict_root)
    create_cand = cc.Candidates(dict_root)
    top_k: int
        Establishes the number (k) of best predicted sentences.
    Methods
    ----------
    __prepare_inputs_model__(sentences, labels = None)
        This method is used to prepare the input sentences as stablished by the BERT model. 
    __predict_type__(prediction_dataloader)
        Return the label and its probability of being this type (whether pausible(1) or impausible(0)).
    __save_plausible_candidates__(b_input_ids, prob, correct_probs, prob_idx_list)
        Save the candidate_sentences which are predicted as 'plausible' and the probability of belonging to that classification.
    __get_predictions_candidates__(prediction_dataloader)
        Predict the label for each candidate_sentences, the probability of belonging to that label and 
        return the candidate_sentences from highest to lowest probability.
    __check_sentence_whether_misspellings__(sentence)
        First step to verify if an input sentence is classified as plausible. Otherwise, the sentence is further analyzed.
    __get_top_k_final_candidates__(correct_probs)
        Return the top k candidate_sentences where k is an integer higher than 1.
    __get_candidate_max_prob__(sentences_candidates)
        Save and return the (top_k) candidate_sentence(s) after having analyzed all the candidates_sentences
    __get_best_candidate_sentences__(sentence,start, label = None)
        Create a set of sentences which have possible corrected spellings (candidates) and return the best candidate_sentence
        with the highest probability
    get_final_suggested_sentences(org_text, label=None)
        The input sentence is analyzed if no misspellings are found,return the input sentence. Otherwise, for each misspelling found 
        in the input sentence obtain candidates 
    return_best_sentence(sentence)
        Obtain the (top_k) best suggested candidate sentence(s) based on the input sentence
    """
    def __init__(self, root_pretrained, checkpoint_dir, dict_root, output_dir, top_k = 1):
        """
        Parameters
        -----------
        root_pretrained:str
            location of the folder where the pretrained clinicalBERT model is located.
        checkpoint_dir:str
            location of the folder where the last checkpoint of SpellChecker model is located. 
        dict_root:str
            location of the folder where the dictionary is located --> used for initializing the PyEnchat Dict.
        output_dir:str
            location of the folder where the results are going to be saved. 
        using_mask(Optional):bool
            Default False; whether the Masked Language Model(MLM) is used. 
        top_k(Optional):int
            Default 1; indicates the number (k) of best predicted sentences.
        """
        self.out_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(root_pretrained, do_lower_case=True)
        self.c_model = load_model_detection(root_pretrained, checkpoint_dir) 
        #if using_mask:
        #    self.ml_model = mlm.MaskLanguageModel(root_pretrained, dict_root)
        self.create_cand = cc.Candidates(dict_root)
        #self.using_mask = using_mask
        self.top_k = top_k
    
    def __prepare_inputs_model__(self,sentences, labels = None):
        """
        This method is used to prepare the input sentences as stablished by the BERT model
        -Tokenization
        -Add special characters
        -Change tokens to indexes based of the vocabulary of clinicalBERT
        -We obtain a torch object.
        Parameters
        ----------
        sentences:list
            list of candidate_sentences or input sentences
        labels(Optional):str
            Default None; whether the labels are known.
        Return
        ------
        torch.utils.data.DataLoader
            input is ready for the SpellChecker model which is based on BERT model.
        """
        if labels:
            data_loader = prepare_dataloader(self.tokenizer,sentences, labels)
        else:
            data_loader = prepare_inputs_dataloader(self.tokenizer,sentences)
        return data_loader

    def __predict_type__(self,prediction_dataloader):
        """
        Verify whether the input is one of these two labels:
        -Pausible or 1: is probable to be grammatically correct. 
        -Impausible or 0: has at least one misspelling.
        Return the label and its probability of being this type.

        Parameters
        ----------
        prediction_dataloader:torch.utils.data.DataLoader
            input data for the SpellChecker model which is based on BERT model.
        Return
        ------
        int
            1 or pausible: sentence has not any misspellings 
            0 or impausible: sentence has one or more misspellings
        float
            probability of the type of prediction (pausible or impausilble)
        """
        predictions = []
        probabilities = []
        for i,batch in enumerate(prediction_dataloader):#tqdm(prediction_dataloader):
            if len(prediction_dataloader) >33:
                print("\rFold sentences {}/{}.".format(i, len(prediction_dataloader)), end='\r')
                sys.stdout.flush()
            # Add batch to GPU#batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids = batch
            if isinstance(b_input_ids, list):
                b_input_ids = b_input_ids[0]
            b_attention_masks = create_attention_mask(b_input_ids)
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = self.c_model(b_input_ids, token_type_ids=None, attention_mask=b_attention_masks)[0]
            probabilities.append(torch.softmax(logits, dim=1).tolist())
            logits = logits.detach().cpu().numpy()
            # Store predictions and true labels
            predictions.append(logits)
        
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        probabilities = probabilities[0][0]
        prob = 0.0
        if flat_predictions[0] == 0:
            prob = probabilities[0]
        else:
            prob = probabilities[1]
        return flat_predictions[0], prob

    def __save_plausible_candidates__(self,b_input_ids, prob, correct_probs, prob_idx_list):
        """
        Append those candidate_sentences which are predicted as 'plausible' 
        and their probabilities of being labeled as 'plausible'

        Parameters
        ----------
        b_input_ids: torch.Tensor
            candidate_sentences which are the replacement of misspellings by the candidates 
        prob: list
            list of probabilities of being classified as 'plausible' or 'implausible'
        correct_probs: pandas.core.frame.DataFrame
            appending the candidate_sentences classified as 'plausible' and their probabilities
        prob_idx_list: array
            the index of the label. 0 is for 'implausible' and 1 is for 'plausible'
        Return
        ------
        pandas.core.frame.DataFrame
            the candidate_sentences classified as 'plausible' and their probabilities
        """
        for i,prob_idx in enumerate(prob_idx_list):
            if prob_idx == 1:
                #print(self.tokenizer.decode(b_input_ids[i].numpy(), skip_special_tokens=True ), 'probability'+str(prob[i][1]))
                correct_probs = correct_probs.append({'candidate':b_input_ids[i].numpy(), 'probability': prob[i][1]}, ignore_index=True)
        correct_probs = correct_probs.sort_values(by=['probability'], ascending=False).reset_index(drop=True)
        return correct_probs

    def __get_predictions_candidates__(self,prediction_dataloader):
        """
        Get the predictions for each candidate_sentences which are sentences with candidates words obtained from the lexicon.
        These candidates are suggestions for the misspellings which are found in the input sentence.
        
        Predict the label for each candidate_sentences, the probability of belonging to that label and return the candidate_sentences 
        return the candidate_sentences from highest to lowest probability.
        Parameters
        ----------
        prediction_dataloader: torch.utils.data.dataloader.DataLoader
        
        Return
        ------
        pandas.core.frame.DataFrame
            correct_probs which are the predicted (as plausible) candidate_sentences ordered from highest to lowest probability.
        """
        correct_probs = pd.DataFrame(columns=['candidate', 'probability'])
        for i,batch in enumerate(prediction_dataloader):#tqdm(prediction_dataloader):
            if len(prediction_dataloader) >33:
                print("\rFold sentences {}/{}.".format(i, len(prediction_dataloader)), end='\r')
                sys.stdout.flush()
            # Add batch to GPU#batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids = batch
            if isinstance(b_input_ids, list):
                b_input_ids = b_input_ids[0]
            b_attention_masks = create_attention_mask(b_input_ids)
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = self.c_model(b_input_ids, token_type_ids=None, attention_mask=b_attention_masks)[0]
            prob = torch.softmax(logits, dim=1).tolist()
            prob_idx = np.argmax(prob, axis=1)
            correct_probs = self.__save_plausible_candidates__(b_input_ids, prob, correct_probs, prob_idx)
        return correct_probs

    def __check_sentence_whether_misspellings__(self,sentence):
        """
        First step to verify if an input sentence is classified as plausible. Otherwise, the sentence is further analyzed.
        Prepare sentence as a proper input for the Spell Checker model based on the requirements
        for the input of the pre-trained BERT model.
        Verify the label of the sentences and its probability of being classified as the obtained label.

        Parameters
        ----------
        sentence:str
            input string to be verified.
        Return
        ------
        int
            1 or pausible: sentence has not any misspellings 
            0 or impausible: sentence has one or more misspellings
        float
            probability of the type of prediction (pausible or impausilble)
        """
        data_loader = self.__prepare_inputs_model__([sentence])
        predictions, probabilities = self.__predict_type__(data_loader) #, _ = self.get_predictions(self.c_model, data_loader)
        return predictions, probabilities 
    
    def __get_top_k_final_candidates__(self,correct_probs):
        """
        Return the top k candidate_sentences where k is an integer higher than 1.

        Parameters
        ----------
        correct_probs:pandas.core.frame.DataFrame
            all candidates sentences obtained from the 
        Return
        ------
        pandas.core.frame.DataFrame
            top_k best candidates sentences with their probabilities
        """
        final_candidates = pd.DataFrame(columns=['candidate', 'probability'])
        if len(correct_probs) < self.top_k:
            self.top_k = len(correct_probs)
        top_k = correct_probs.probability.astype(float).nlargest(self.top_k)#.reset_index(drop=True)
        top_k_idx = top_k.index#.item()
        for i,tk in enumerate(top_k):
            sen_can = self.tokenizer.decode(correct_probs.candidate.iloc[top_k_idx[i]], skip_special_tokens=True )
            final_candidates = final_candidates.append({'candidate':sen_can, 'probability': tk}, ignore_index=True)
        return final_candidates

    def __get_candidate_max_prob__(self,candidates_sentences):
        """
        1.-Prepare the candidates_sentences based on the specifications for the inputs of BERT model.
        2.-Get predictions of the candidates_sentences 
        3.-Obtain the ordered form higher to lower probability candidates_sentences which are predicted as 'plausible'(1)  
        4.-Save and return the (top_k) candidate_sentence(s) after having analyzed all the candidates_sentences

        Parameters
        ----------
        candidates_sentences: torch.utils.data.DataLoader
            the list of new sentences which have the suggested candidates for each misspelling
        Return
        ------
        pandas.core.frame.DataFrame
            best candidate_sentence or top k candidate_sentences with their probabilities
        """
        data_loader = self.__prepare_inputs_model__(candidates_sentences)
        correct_probs = self.__get_predictions_candidates__(data_loader)
        final_candidates = pd.DataFrame(columns=['candidate', 'probability'])
        if not correct_probs.empty:
            if self.top_k != 1:
                final_candidates = self.__get_top_k_final_candidates__(correct_probs)
            else:
                sen_can = self.tokenizer.decode(correct_probs.candidate.iloc[0], skip_special_tokens=True )
                final_candidates = final_candidates.append({'candidate':sen_can, 'probability': correct_probs.probability.astype(float)[0]}, ignore_index=True)
        return final_candidates

    def __get_best_candidate_sentences__(self,sentence,label = None):
        """
        Create a set of sentences which have possible corrected spellings (candidates)
        Analyze all candidates_sentences
        Return a dictionary which contains
            -sentences:input sentence, 
            -suggested:best candidates sentences or top k candiates_sentences,
            -probability: the probability of being plausible,
            -label:the truth label (None if there is no label) 
        Parameters
        ----------
        sentence:list
            words of the input sentences
        label:str
            (Optional) the correct form of the sentence
        Return
        ------
        dict
            dictionary which contains the original sentence, the suggested sentence with the best or top k candidates, the probability and the label
        """
        candidates_sentences,_ =  self.create_cand.get_candidate_sentence(sentence)
        candidates_sentences = [' '.join(candidate) for candidate in candidates_sentences]
        #print('CANDIDATES_SENTENCES: ',candidates_sentences)
        if len(candidates_sentences) > 200:
            print(sentence)
            print('Len of candidates:{} '.format(len(candidates_sentences)))
        # if len(candidates_sentences) > 1000: ########## HERE a problem?###########
        #     candidates_sentences = candidates_sentences[:500]    
        if not candidates_sentences:
            final_dict = {'sentences':sentence, 'suggested':['No suggestions'], 'probability': [0.0], 'label':label}
        else:
            candidates_sentences = self.__get_candidate_max_prob__(candidates_sentences)
            final_dict = {'sentences':' '.join(sentence), 'suggested':candidates_sentences.candidate.values, 
                            'probability': candidates_sentences.probability.values, 'label':label}
        return final_dict

    def get_final_suggested_sentences(self,org_text,final_output, label=None):
        """
        The input sentences goes to the following steps:
            -lowercase
            -clean_sentence: remove any url, extra spaces, and symbols 
            -predict whether the sentence is plausible or implausible. The first means, the sentence has a high probability of containing no misspellings.
                The latter means, the sentence has (a) misspelling(s).
            -If the last step has a probability lower than a threshold (e.g. 0.95), considered it as having at least one misspelling.
                --Get the best top k sentences based on the candidates for each misspelling found in the sentence.
                --These candidates are obtained from a lexicon.
            -Otherwise
                --save the original sentence
        Parameters
        ----------
        org_text: str
            the input sentences
        label: str
            (Optional) the correct form of the sentence
        Return
        ------
        pandas.core.frame.DataFrame
            final output which contains the orginal sentence, the suggested candidate sentence (if misspellings are found),
            the probability and the label (Optional).
        """
        start = time.time()
        text = org_text.lower()
        text = clean_sentence(text)
        type_, prob = self.__check_sentence_whether_misspellings__(text) 
        
        if type_ == 0 or prob < 0.95:
            text = text.strip().split()
            ############ candidates from dictionary ###########
            final_dict = self.__get_best_candidate_sentences__(text,label)
        else:
            #### Sentence labelled as plausible #####
            final_dict = {'sentences':text, 'suggested':org_text, 'probability':prob, 'label':label}
        final_output = append_results(final_output,final_dict, start)
        return final_output

    def return_best_sentence(self, sentence):
        """
        Obtain the (top_k) best suggested candidate sentence(s) based on the input sentence.
        This method is for the data of KOEHLER
        Parameters
        ----------
        sentence:str
            the input sentence
        Return
        ------
        arrays
            the best suggested sentence with its score and the total execution time  
        """
        final_output = pd.DataFrame(columns=['sentences', 'suggested', 'probability','label', 'time'])
        final_out = self.get_final_suggested_sentences(sentence,final_output)
        bestSentence = final_out.suggested.values#[:self.top_k]
        if not bestSentence:
            bestSentence = sentence
        bestScore = final_out.probability.values#[:self.top_k]
        final_time = final_out.time.values
        if not isinstance(bestSentence,str):
            bestSentence = bestSentence[0]
        return bestSentence, bestScore, final_time          
