# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""This code is heavily adapted from the huggingface example on question-answering task (available at: https://github.com/huggingface/transformers/blob/master/examples/legacy/question-answering/run_squad.py) with some modifications."""


import sys
sys.path.append('../')

import warnings
warnings.filterwarnings("ignore")

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from tokenize_dataset import load_and_cache_examples
import pickle, random, itertools, collections, json, pickle
import argparse, glob, logging, random, timeit
import re, string
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import transformers
from transformers import MODEL_FOR_QUESTION_ANSWERING_MAPPING, WEIGHTS_NAME, AdamW, AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup, squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.data.metrics.squad_metrics import compute_predictions_logits, compute_predictions_log_probs, squad_evaluate
from transformers.trainer_utils import is_main_process

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
        
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

print(f'{torch.cuda.device_count()} GPU found')


def normalize_answer(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def get_clean_ans(s):
    if not s:
        return []
    return normalize_answer(s)

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same/len(pred_toks)
    recall = 1.0 * num_same/len(gold_toks)
    f1 = (2 * precision * recall)/(precision + recall)
    
    return f1

def compute_precision(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same/len(pred_toks)
    
    return precision

def compute_recall(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    
    recall = 1.0 * num_same/len(gold_toks)
    
    return recall

def get_raw_scores(examples, predictions):
    predictions = dict(predictions)
    preds = {}
    for k,v in predictions.items():
        preds[str(k)] = v
        
    exact_scores = {}
    f1_scores = {}
    recall_scores = {}
    precision_scores = {}
    
    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"]
                        for answer in example.answers
                        if normalize_answer(answer["text"])]
        if not gold_answers:
            gold_answers = [""]
            
        if str(qas_id) not in preds:
            print(f"Missing prediction for {qas_id}")
            continue
            
            
        qas_id = str(qas_id)
        prediction = preds[qas_id]
        exact_scores[qas_id]     = max(compute_exact(a,prediction) for a in gold_answers)
        f1_scores[qas_id]        = max(compute_f1(a,prediction) for a in gold_answers)
        precision_scores[qas_id] = max(compute_precision(a,prediction) for a in gold_answers)
        recall_scores[qas_id]    = max(compute_recall(a,prediction) for a in gold_answers)
        
    return exact_scores, f1_scores, recall_scores, precision_scores

    
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def to_list(tensor):
    return tensor.detach().cpu().tolist()

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1,0] and not evaluate:
        torch.distributed.barrier()
        
    input_dir = args.cache_dir if args.cache_dir else "."
            
    if args.do_train:
        prefix = "train"
    elif args.do_evaluate:
        prefix = "test"
    
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}_{}".format(
            "train" if args.do_train else "dev",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            prefix,
            args.dataset_name,
        ),
    )
    
    print(cached_features_file)
    
    logger.info("Loading features from cached file %s", cached_features_file)
    features_and_dataset = torch.load(cached_features_file)
    features, dataset, examples = (
        features_and_dataset['features'],
        features_and_dataset['dataset'],
        features_and_dataset['examples'],
    )
            
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()
        
    if output_examples:
        return dataset, examples, features
    
    return dataset


def train(args, train_dataset, model, tokenizer):
    if args.local_rank in [-1,0]:
        tb_writer = SummaryWriter()
        
    args.train_batch_size = args.per_gpu_train_batch_size*max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps*args.num_train_epochs
        
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps = t_total
    )
    
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
        
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        
    logger.info("***** Running Training *****")
    logger.info(" Num examples = %d", len(train_dataset))
    logger.info(" Num epochs = %d", args.num_train_epochs)
    logger.info(" Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        " Total train batch size (w, parallel, distributed & accumulation) = %d",
        args.train_batch_size*args.gradient_accumulation_steps*(torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info(" Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info(" Total optimization steps = %d", t_total)
    
    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    
    if os.path.exists(args.model_name_or_path):
        try:
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
            
            logger.info(" Continuing training from checkpoint, will skip to saved global_step")
            logger.info(" Continuing training from epoch %d", epochs_trained)
            logger.info(" Continuing training from global step %d", global_step)
            logger.info(" Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info(" Starting fine-tuning")
            
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable = args.local_rank not in [-1, 0]
    )
    
    set_seed(args)

    em_list = []
    f1_list = []
    re_list = []
    pr_list = []
    
    ite = 0
    for _ in train_iterator:
        ite+=1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
                
            model.train()
            
            batch = tuple(t.to(args.device) for t in batch)
            
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }
            
            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]
            
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index":batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64)*args.lang_id).to(args.device)}
                    )
                    
                    
            outputs = model(**inputs)
            loss = outputs[0]
            
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                
                
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                
                if args.local_rank in [-1,0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:
                        reults = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_ls()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                    
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_model_dir, "checkpoint-{}".format(global_step))
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(model_to_save.state_dict(), f'{output_dir}/model.pt')
                    tokenizer.save_pretrained(output_dir)
                    
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break


        if ite>=0:
            
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(args.output_model_dir)
            torch.save(model_to_save.state_dict(), f'{args.output_model_dir}/model_ite{ite}.pt')
            tokenizer.save_pretrained(args.output_model_dir)

            torch.save(args, os.path.join(args.output_model_dir, f"training_args_ite{ite}.bin"))
            logger.info("Saving model checkpoint to %s", args.output_model_dir)

            torch.save(optimizer.state_dict(), os.path.join(args.output_model_dir, f"optimizer_ite{ite}.pt"))
            torch.save(scheduler.state_dict(), os.path.join(args.output_model_dir, f"scheduler_ite{ite}.pt"))
            logger.info("Saving optimizer and scheduler states to %s", args.output_model_dir)

        em_ls, f1_ls, re_ls, pr_ls = evaluate(args, model, tokenizer, TEST_DURING_EPOCH=True)
        em_list.append(np.mean(list(em_ls.values())))
        f1_list.append(np.mean(list(f1_ls.values())))
        re_list.append(np.mean(list(re_ls.values())))
        pr_list.append(np.mean(list(pr_ls.values())))
        
        print(em_list)
        print(f1_list)
        print(re_list)
        print(pr_list)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    
    with open(args.output_dir + "exact_match.pkl", "wb") as f:
        pickle.dump(em_list, f)
    with open(args.output_dir + "f1_score.pkl", "wb") as f:
        pickle.dump(f1_list, f)
    with open(args.output_dir + "precision.pkl", "wb") as f:
        pickle.dump(pr_list, f)
    with open(args.output_dir + "recall.pkl", "wb") as f:
        pickle.dump(re_list, f)
    
    
    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)
        
    if args.local_rank in [-1, 0]:
        tb_writer.close()
        
    return global_step, tr_loss / global_step
                    
    
def load_and_cache_test_examples(args, tokenizer, evaluate=True, output_examples=True):
    input_dir = args.cache_dir if args.cache_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}_{}".format(
            "dev",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            "val",
            args.dataset_name,
        ),
    )
    
    print(cached_features_file)
    
    logger.info("Loading features from cached file %s", cached_features_file)
    features_and_dataset = torch.load(cached_features_file)
    features, dataset, examples = (
        features_and_dataset['features'],
        features_and_dataset['dataset'],
        features_and_dataset['examples'],
    )

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()
        
    if output_examples:
        return dataset, examples, features
    
    return dataset

def evaluate(args, model, tokenizer, TEST_DURING_EPOCH=False, prefix=""):
    if TEST_DURING_EPOCH:
        dataset, examples, features = load_and_cache_test_examples(args, tokenizer, evaluate=True, output_examples=True)
    else:
        dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
        
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
        
    eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)
    
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
        
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info(" Num examples = %d", len(dataset))
    logger.info(" Batch size = %d", eval_batch_size)
    
    all_results = []
    start_time = timeit.default_timer()
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            
            if args.model_type in ["xlm", "roberta", "distilbert", "camembert",
                                   "bart", "longformer"]:
                del input["token_type_ids"]
                
            feature_indices = batch[3]
            
            if args.model_type in ["xlnet","xlm"]:
                inputs.update({"cls_index":batch[4], "p_mask":batch[5]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64)*args.lang_id).to(args.device)}
                    )
            outputs = model(**inputs)
            
        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            
            if args.model_type in ["bart"]:
                output = [to_list(output[i]) for output in outputs[:2]]
            else:
                output = [to_list(output[i]) for output in outputs.to_tuple()]
                
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]
                
                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits
                )
            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                
            all_results.append(result)
            
            
    evalTime = timeit.default_timer() - start_time
    logger.info(" Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odd_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None
        
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top
        
        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )
        
    results = squad_evaluate(examples, predictions)
    
    print(results)
    
    em, f1, recall, precision = get_raw_scores(examples, predictions)
    assert list(em.keys()) == list(f1.keys()) == list(precision.keys()) == list(recall.keys())

    if TEST_DURING_EPOCH == False:
        
        df = pd.DataFrame()
        df['qas_id'] = list(em.keys())
        df['EM'] = list(em.values())
        df['F1'] = list(f1.values())
        df['Recall'] = list(recall.values())
        df['Precision'] = list(precision.values())

        df_file_name = os.path.join(
            args.output_dir,
            "results_{}_{}_{}.csv".format(
                list(filter(None, args.orig_model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(args.dataset_name),
            ),
        )
        df.to_csv(df_file_name, index=False)
        
        return results
    
    return em, f1, recall, precision


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--orig_model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--output_model_dir", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--predict_file", default=None, type=str)
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--version_2_with_negative", action="store_true")
    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--doc_stride", default=128, type=int)
    parser.add_argument("--max_query_length", default=20, type=int)
    parser.add_argument("--max_answer_length", default=100, type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_evaluate", action="store_true")
    parser.add_argument("--do_validate", action="store_true")
    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int) ##
    parser.add_argument("--per_gpu_eval_batch_size", default=256, type=int) ##
    parser.add_argument("--learning_rate", default=3e-5, type=float) ##
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=5.0, type=float) ###########
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--n_best_size", default=20, type=int)
    parser.add_argument("--verbose_logging", action="store_true")
    parser.add_argument("--lang_id", default=0, type=int)
    parser.add_argument("--logging_steps", default=500000, type=int)
    parser.add_argument("--save_steps", default=500000, type=int)
    parser.add_argument("--eval_all_checkpoints", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", default="01", type=str)
    parser.add_argument("--server_ip", default="", type=str)
    parser.add_argument("--server_port", default="", type=str)
    parser.add_argument("--threads", default=64, type=int)
    parser.add_argument("--dataset_name", default="IDU", type=str)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--overwrite_output_model_dir", action="store_true")
    parser.add_argument("--which_epoch", default=0, type=int)
    
    args = parser.parse_args()
    
    print(f'{torch.cuda.device_count()} GPU found')
    
    
    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superios to the document length in examples."
        )
        
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory already exists and is not empty. Use --overwrite_output_dir"
        )
        
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_progress_group(backend="nccl")
        args.n_gpu = 1
        
    args.device = device
    
    print("args.device: ", args.device)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1,0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        
    set_seed(args)
    
    if args.local_rank not in [-1,0]:
        torch.distributed.barrier()
        
    args.model_type = args.model_type.lower()
    
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        #args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        #do_lower_case=args.do_lower_case,
        #use_fast=False,
    #)
    
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path)

    if args.local_rank == 0:
        torch.distributed.barrier()
        
    model.to(args.device)
    print("Loading trained model . . . .")
    model.load_state_dict(torch.load(f'{args.output_model_dir}/model_ite{args.which_epoch}.pt')) ##################
    
    logger.info("Training/evaluation parameters %s", args)
        
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info("global_step = %s, average_loss = %s", global_step, tr_loss)
        
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not os.path.exists(args.output_model_dir):
            os.makedirs(args.output_model_dir)
            
        logger.info("Saving model checkpoint to %s", args.output_model_dir)
        
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), f'{args.output_model_dir}model.pt')
        model_to_save.save_pretrained(args.output_model_dir)
        tokenizer.save_pretrained(args.output_model_dir)
        
        torch.save(args, os.path.join(args.output_model_dir, "training_args.bin"))
        
        model = AutoModelForQuestionAnswering.from_pretrained(args.output_model_dir)
        
        tokenizer = AutoTokenizer.from_pretrained(args.output_model_dir, do_lower_case=args.do_lower_case, use_fast=False)
        
        model.to(args.device)
        
    results = {}
    if args.do_evaluate: #and args.local_rank in [-1,0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_model_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_model_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
        else:
            checkpoints = [args.output_model_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_model_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
            

            logger.info("Evaluate the following checkpoints: %s", checkpoints)

            for checkpoint in checkpoints:
                result = evaluate(args, model, tokenizer, prefix = args.dataset_name)

                print('RESULT: ', dict(result))

    return result
        
        
if __name__ == "__main__":
    main()
       
            
    
        
            
    
    