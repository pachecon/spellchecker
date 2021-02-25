import enchant
import argparse
import logging

import getNonWordTypos as non_errors
import getRealWordTypos as real_errors 

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

def main():
    parser.add_argument("--data_dir",
                      default='../../data/train/',
                      type=str,
                      required=False,
                      help="The input data dir. where the files as csv file are located.")

    parser.add_argument("--dict_root",
                      default='../../dictionaries/',
                      type=str,
                      required=False,
                      help="The directory of the dictionaries")
    
    parser.add_argument("--output_dir",
                      default='../../data/misspelled_corpora/',
                      type=str,
                      required=False,
                      help="The directory where the results will be saved and/or updated.")
    
    args = parser.parse_args()

    for type_ in ['train','val','test']:
        fname = args.data_dir+type_+'_data.csv'
        nontypos = non_errors.NonWordTypos(type_,fname,dict_root=args.dict_root,output_root=args.output_dir)
        nontypos.create_edit_misspellings()
        realtypos= real_errors.RealWordTypos(type_,fname,dict_root=args.dict_root,output_root=args.output_dir)
        realtypos.word_errors()

    
if __name__ == "__main__":
    main()