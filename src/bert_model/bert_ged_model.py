import argparse
import io
import logging
import numpy as np
import pandas as pd
import prepare_sentences_labels
import sys
import torch

from pytorch_pretrained_bert import BertAdam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer 
from utils_bert import prepare_dataloader, flat_accuracy, load_last_saved_checkpoint, create_attention_mask,save_checkpoints

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
MAX_LEN = 128 # Set the maximum sequence length. In the original paper, the authors from BERT used a length of 128 or  512.
BATCH_SIZE = 32 # Select a batch size for training. For fine tuning BERT on a specific task , BERT authors recommend a batch size of 16 or 32

def main():
  parser.add_argument("--data_dir",
                      default=None,
                      type=str,
                      required=False,
                      help="The input data dir. as csv file.")
  
  parser.add_argument("--checkpoint_dir",
                      default="./save_ckpt/",
                      type=str,
                      required=False,
                      help="The directory to save the checkpoint or where the last checkpoint was saved.")

  parser.add_argument("--pretrain_dir",
                      default='../pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/',
                      type=str,
                      required=False,
                      help="The directory where the clinicBERT is located.")

  parser.add_argument("--do_train",
                      action='store_true',
                      help="Whether to run training.")
  
  parser.add_argument("--do_eval",
                      action='store_true',
                      help="Whether to run eval on the dev set.")
  
  parser.add_argument("--learning_rate",
                      default=2e-5,
                      type=float,
                      help="The learning rate for Adam optimizer.")
  parser.add_argument("--num_epochs",
                      default=3.0,
                      type=float,
                      help="Number of epochs.")
  parser.add_argument("--warmup_proportion",
                      default=0.1,
                      type=float,
                      help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")
  parser.add_argument("--continue_training",
                      default=False,
                      type=bool,
                      help="Useful when we want to continue training from the last saved checkpoint.")
  parser.add_argument("--last_epoch",
                      default=0.0,
                      type=float,
                      help="Give the last epoch where the model stayed.")
  

  args = parser.parse_args()

  root_pretrained = args.pretrain_dir
  print('Load Tokenizer from clinicalBERT...')
  tokenizer = AutoTokenizer.from_pretrained(root_pretrained, do_lower_case=True)#'monologg/biobert_v1.1_pubmed')#'gsarti/biobert-nli')#)
  print('Load pretrained model clinicalBERT... ')
  model = AutoModelForSequenceClassification.from_pretrained(root_pretrained)#, from_pt=True)
  epoch = 0
  steps = 0
  loss_saved = 0
  #Prepare data for training
  if args.do_train:
    print('---------------------------Prepare data for training-----------------------') 
    if not args.data_dir:
      #Use default datasets for training
      sentences_train, labels_train = prepare_sentences_labels.prepare_data_train()
      #print(len(sentences_train))
    else:
      #The input data should have a column of labels and a column of sentences
      try:
        sentences_train, labels_train = prepare_sentences_labels.prepare_data_train_val(args.data_dir+'data_train.csv') 
      except:
        print('Error by loading the file for training. Either the name file is incorrect (should be named data_train.csv) or the file has no the structure of labels, sentences')
        print('Please check your file')
        exit()
    train_dataloader = prepare_dataloader (tokenizer, sentences_train, labels_train)
  
  #Prepare data for validation 
  if args.do_eval:
    print('---------------------------Prepare data for validation-----------------------') 
    if not args.data_dir:
      #Use default datasets for validation
      sentences_val, labels_val = prepare_sentences_labels.prepare_data_val()
    else:
      #The input data should have a column of labels and a column of sentences
      try:
        sentences_val, labels_val = prepare_sentences_labels.prepare_data_train_val(args.data_dir+'data_val.csv') 
      except:
        print('Error by loading the file for training. Either the name file is incorrect (should be named data_val.csv) or the file has no the structure of labels, sentences')
        print('Please check your file')
        exit()
    val_dataloader = prepare_dataloader (tokenizer,sentences_val, labels_val)
  print('---------------------------------------------------------------------------')
  if args.do_train: 
    print('Hyperparameters for training and for Adam optimization are loading...')
    train_total_steps = len(train_dataloader) * args.num_epochs
    # Hyperparameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.00}]
    
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion, t_total=train_total_steps)
    if args.continue_training:
        model, epoch, loss_saved, steps, optimizer = load_last_saved_checkpoint(args.checkpoint_dir, args.last_epoch, model, optimizer)
  epochs = np.arange(args.num_epochs)

  for ep, _ in enumerate(epochs, start=epoch):
    ################
    # Training
    ################
    if args.do_train:
      # Set our model to training mode (as opposed to evaluation mode)
      model.train()
      print('Training is going to start...')    
      # Tracking variables
      train_loss = 0 
      number_train_steps = 0 
      loss_total_train = []

      if steps != 0:
        train_loss = loss_saved
      logger.info("***** Running training *****")
      logger.info("  Num examples = %d", len(train_dataloader))
      logger.info("  Batch size = %d", BATCH_SIZE)
      logger.info("  Num steps = %d", train_total_steps)
      
      # Train the data for one epoch
      for step, batch in enumerate(tqdm(train_dataloader)):
        # Add batch to GPU: batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_labels = batch
        b_input_mask = create_attention_mask(b_input_ids)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # Forward pass: compute predicted outputs and calculate the batch loss
        loss = model(b_input_ids, token_type_ids=None, 
                    attention_mask=b_input_mask, labels=b_labels)[0]
        # Backward pass:: compute gradient of the loss with respect to model parameters
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        # Update tracking variables
        train_loss += loss.item()
        #nb_tr_examples += b_input_ids.size(0)
        number_train_steps += 1

        if step%100 == 0:
          #Save the checkpoints every n_btach size of 100 
          save_checkpoints(ep, model, optimizer, train_loss, step, args.checkpoint_dir)
        loss_total_train.append(train_loss)
      save_checkpoints(ep, model, optimizer, train_loss, len(train_dataloader), args.checkpoint_dir)
      print("Train loss: {}".format(train_loss/number_train_steps))
      df = pd.DataFrame(loss_total_train)
      df.to_csv(args.checkpoint_dir+'loss_train_'+str(ep)+'.csv', index=False)

    ################
    # Validation
    ################
    if args.do_eval:   
      print('Prepare validation')
      model.eval()
      # Tracking variables
      eval_loss, eval_accuracy = 0, 0
      number_eval_steps= 0
      loss_total_val, acc_total_val = [], []

      logger.info("***** Running evaluation *****")
      logger.info("  Num examples = %d", len(val_dataloader))
      logger.info("  Batch size = %d", BATCH_SIZE) # batch size can be different for validation and training but for the moment they are the same
      exit()
      # Evaluate data for one epoch
      for batch in tqdm(val_dataloader):
          # Add batch to GPU: batch = tuple(t.to(device) for t in batch)
          # Unpack the inputs from our dataloader
          b_input_ids, b_labels = batch
          b_input_mask = create_attention_mask(b_input_ids)

          # The gradients are neither computed nor stored.
          with torch.no_grad():
              # Forward pass, calculate logit predictions
              logits = model(b_input_ids, token_type_ids =None, attention_mask=b_input_mask)[0]
          
          loss = CrossEntropyLoss() #weight=[0.8, 1.2, 0.97]
          tmp_eval_loss = loss(logits, b_labels)
          logits = logits.detach().cpu().numpy()
          label_ids = b_labels.to('cpu').numpy()
          eval_loss += tmp_eval_loss
          tmp_eval_accuracy = flat_accuracy(logits, label_ids)
          eval_accuracy += tmp_eval_accuracy
          number_eval_steps += 1
          loss_total_val.append(eval_loss)
          acc_total_val.append(eval_accuracy)
      print("Validation Accuracy: {}".format(eval_accuracy/number_eval_steps))
      df = pd.DataFrame(loss_total_val)
      df.to_csv(args.checkpoint_dir+'loss_val_'+str(ep)+'.csv', index=False)
      df = pd.DataFrame(acc_total_val)
      df.to_csv(args.checkpoint_dir+'acc_val_'+str(ep)+'.csv', index=False)

  print('Training is over!')
  torch.save(model.state_dict(), args.checkpoint_dir+'bert-GD.pth')
  print('Model is saved as {}bert-GD.pth'.format(args.checkpoint_dir))

if __name__ == "__main__":
  main()