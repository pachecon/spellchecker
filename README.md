<<<<<<< HEAD
## Table of contents

- [Table of contents](#table-of-contents)
- [Quick start](#quick-start)
  - [Main file `main_koehler.py`](#main-file-rsdiabetespy)
  - [Install libraries](#install-libraries)
  - [Run project](#run-project)
  - [Run only the file regarding the noisy channel model](#run-only-the-file-regarding-the-noisy-channel-model)
  - [Run only the file regarding the Spell Checker](#run-only-the-file-regarding-the-Spell-Checker)
  - [Run only the file regarding the Misspelling Generator](#run-only-the-file-regarding-the-misspelling-generator)
- [Description](#description)
- [What's included](#whats-included)
  - [Author](#author)
  - [Version](#version)

## Quick start
This project is focused on solving misspellings in medical terms and drug names. 
The Spell Checker model is fine-tuned based on the pre-trained clinicalBERT model. 

### Main file `main_koeheler.py`

### Install libraries

- It is needed to install the libraries mentioned on requirements.txt
  ```
  pip3 install -r requirements.txt
  ```
- If there is an error for installing library pyFM, then please try:
  ```
  pip3 install git+https://github.com/coreylynch/pyFM
  ```
source: https://github.com/coreylynch/pyFM/blob/master/README.md

### Run project

- Run the main file
  ```
  python3 main_koehler.py --input_dir=../data/test/ --model_type=bert_model --checkpoint_dir=./bert_model/save_ckpt/ --root_pretrained=./pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/ --dict_root=../dictionaries/ --output_dir=../output/results/ --top_k=1 --db_host=localhost --db_user=arlette --db_passwd=arlette 
  ```
- main_koehler:
  - input_dir: The input data directory where the folders and files as csv file are located.
  - model_type
    - **bert_model**: Spell Checker model
    - **noisy_model**: Baseline Noisy Channel Model
  - dict_root: The directory of the dictionary for the list of candidates and the abbreviations dictionary
  - *for the Spell Checker* (**bert_model**):
    - checkpoint_dir: The directory to save the checkpoint or where the last checkpoint was saved.
    - root_pretrained: The directory of the pre-trained clinicalBERT
    - top_k:
  - *for the Noisy Channel Model* (**noisy_model**):
    - db_user: The user name for getting access to the database for the Noisy Channel Model where the bigrams are located. 
    - db_passwd: The password for getting access to the database for the Noisy Channel Model where the bigrams are located.

### Run only the file regarding the noisy channel model
- Run the main_noisychannelmodel file
  ```
  python3 main_noisychannelmodel.py --data_dir=../../data/misspelled_corpora/ --dict_root=../../dictionaries/ --output_dir=../../output/results/baseline/ --output_dir_train=../../output/train/, --do_train=False --db_host=localhost --db_user=arlette --db_passwd=arlette 
  ```
- main_noisychannelmodel:
  - data_dir: The input data directory where the csv files are located.
  - dict_root: The directory of the dictionary for the candidates 
  - output_dir:The directory where the results will be saved and/or updated.
  - output_dir_train: The directory where the data after training the noisy channel model is saved.
  - do_train: For training
  - db_host: Host database for the Noisy Channel Model.
  - db_user: The user name for getting access to the database for the Noisy Channel Model where the bigrams are located. 
  - db_passwd: The password for getting access to the database for the Noisy Channel Model where the bigrams are located.

### Run only the file regarding the Spell Checker
- Run the mainBERT file:
  ```
  python3 mainBERT.py --data_dir=../../data/misspelled_corpora/ --checkpoint_dir=./save_ckpt/ --root_pretrained=../pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/ --dict_root=../../dictionaries/ --output_dir=../../output/results/bert/
  --top_k=1
  ```
- mainBERT:
  - data_dir: The input data directory where the csv files are located.
  - dict_root: The directory of the dictionary for the list of candidates and the abbreviations dictionary
  - checkpoint_dir: The directory to save the checkpoint or where the last checkpoint was saved.
  - root_pretrained: The directory of the pre-trained clinicalBERT
  - output_dir: The directory where the results will be saved and/or updated.
  - top_k:

### Run only the file regarding the Misspelling Generator
- Run the main_misspelling_gen file:
  ```
  python3 main_misspelling_gen.py --data_dir=../../data/train/ --dict_root=../../dictionaries/ --output_dir=../../data/misspelled_corpora/
  ```
- main_misspelling_gen:
  - data_dir: The input data directory where the csv files are located.
  - dict_root: The directory of the dictionary 
  - output_dir: The directory where the results will be saved and/or updated.

## Description



## What's included

File system architecture

```text
main_folder/
└── data/
│   ├──misspelled_corpora/ 
│   ├──train/ 
│   └──UMLS/
└── dictionaries/
│   └──meddra/
│   └──who/
│   └──abbreviation_list.csv
│   └──meddra_terms.csv
│   └──multilabel_dic_unique_order.csv
│   └──who_dic.csv
└──src/
│   └──baseline/
|   |  └──connect_database.py: To connect into the
|   |     database "ngrams_db" for getting acces to 
|   |     the bigrams
|   |  └──create_database.py: Create the database 
|   |     called "ngrams_db" and the table ngrams2 
|   |     for saving the bigrams
|   |  └──main_noisychannelmodel.py: Example for 
|   |     runing the baseline
|   |  └──noisy_channel_model.py: The algorithm for
|   |     calculating the score for all candidates 
|   |     and return the best sentence with the most 
|   |     probable candidate of being closer to the 
|   |     observed misspelling
|   |  └──utils_ncm.py
|   |     extra functions for other tasks related to |   |     save or load data
│   └──bert_model/
│   │  └──save_ckpt/
|   |  |  └──bert-GD.pth
|   |  |  └──model_3.pt
|   |  └── bert_ged_model.py: for training and 
|   |      validation the Spell Checker model 
|   |  └── spell_checker.py: Primarly file to use 
|   |      the already trained Spell Checker model 
|   |      to verify which sentence has the highest 
|   |      probability of being plausible from a 
|   |      list of candidates_sentences.  
|   |  └── utils_bert.py: extra functions for 
|   |      preparing the input into the BERT format   
|   |  └── mainBERT.py 
|   |  └── prepare_sentences_labels.py
│   └──candidates/
|   |  └──get_candidates.py
│   └──extras/
|   |  └──abbreviations.py
|   |  └──detect_type.py
│   └──misspelling_generator/
|   |  └──getNonWordTypos.py
|   |  └──getRealWordTypos.py
|   |  └──getTyposKB.py
|   |  └──main_misspelling_gen.py
|   |  └──utils_mg.py
│   └──prepare_data/
|   |  └──a-z_order_ngrams.py
|   |  └──connect_umlsDB.py
|   |  └──detect_type.py
│   └──pretrained_bert_tf/
|   |  └──biobert_petrain_output_all_notes_150000/
|   |  |  └──config.json
|   |  |  └──vocab.txt
|   |  |  └──model.ckpt
│   └──main_koehler.py
│   └──utils_koehler.py 
└──output/
│   └──results/
|   |  └──baseline/
|   |  └──bert/
|   |  └──noisy_model/
|   |  └──bert_model/
│   └──train/
|   |  └──baseline_ngrams
|   |  |  └──unigrams_order.csv
|   |  |  └──bigrams_order.csv
pyenv/
```

### Author

<a href="https://github.com/pachecon" target="_blank">Arlette M. Perez Espinoza</a>

### Version

1.0.0
=======
# Spellchecker
>>>>>>> 6563cd2fcca30149b8a2b1441bed503ca81be15c
