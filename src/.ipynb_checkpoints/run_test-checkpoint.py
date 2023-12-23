import os

os.system("python3 finetune_model.py --model_type bert --orig_model_name_or_path clinicalbert --model_name_or_path <PATH TO THE PRE-TRAINED MODEL> --output_dir <PATH TO THE OUTPUT DIRECTORY> --output_model_dir <PATH TO THE TRAINED MODEL DIRECTORY> --data_dir <PATH TO THE DATA DIRECTORY> --cache_dir <PATH TO THE TOKENIZED DATA DIRECTORY> --dataset_name idu --do_lower_case --do_evaluate --which_epoch 4")
