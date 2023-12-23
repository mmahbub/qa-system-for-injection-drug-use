import os

os.system("python3 tokenize_dataset.py --orig_model_name_or_path clinicalbert --model_name_or_path <PATH TO THE PRE-TRAINED MODEL> --output_dir <PATH TO THE TOKENIZED DATA DIRECTORY> --data_dir <PATH TO THE DATA DIRECTORY> --predict_file <FILE NAME>.json --dataset_name idu --do_lower_case --do_train")

# for tokenizing train set use '--do_train'
# for tokenizing test set use '--do_evaluate'
# for tokenizing validation set use '--do_evaluate --do_validate'