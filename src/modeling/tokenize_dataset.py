# this code is heavily adapted from the huggingface example on question-answering task: https://github.com/huggingface/transformers/blob/master/examples/legacy/question-answering/run_squad.py


import argparse
import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import transformers
from transformers import AutoTokenizer, squad_convert_examples_to_features

from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.trainer_utils import is_main_process

logger = logging.getLogger(__name__)


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1,0] and not evaluate:
        torch.distributed.barrier()
        
    input_dir = args.output_dir if args.output_dir else "."
    
    if args.do_train:
        pref = "train"
    elif args.do_validate:
        pref = "val"
    else:
        pref = "test"
        
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}_{}".format(
            "dev" if args.do_evaluate else "train",
            args.orig_model_name_or_path,
            #list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            pref,
            args.dataset_name,
        ),
    )
    
    print(cached_features_file)
    
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset['features'],
            features_and_dataset['dataset'],
            features_and_dataset['examples'],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        
        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file)
        
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride = args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset='pt',
            threads=args.threads,
        )
        
        if args.local_rank in [-1,0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)
            
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()
        
    if output_examples:
        return dataset, examples, features
    
    return dataset
        
        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--orig_model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--predict_file", default=None, type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--version_2_with_negative", action="store_true")
    parser.add_argument("--max_seq_length", default=512, type=int) ##512
    parser.add_argument("--doc_stride", default=128, type=int)
    parser.add_argument("--max_query_length", default=20, type=int)
    parser.add_argument("--max_answer_length", default=100, type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_evaluate", action="store_true")
    parser.add_argument("--do_validate", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--verbose_logging", action="store_true")
    parser.add_argument("--lang_id", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", default="01", type=str)
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--dataset_name", default="IVDU", type=str)
    parser.add_argument("--overwrite_cache", action="store_true")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case = args.do_lower_case,
        cache_dir = args.cache_dir if args.cache_dir else None,
        use_fast= False,
    )
    if args.do_train:
        load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
    elif args.do_evaluate:
        load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)


if __name__ == "__main__":
    main()