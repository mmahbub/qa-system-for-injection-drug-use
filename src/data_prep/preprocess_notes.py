import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
import numpy as np
import pickle
import random
import itertools
import collections
import re
from tqdm import tqdm

from nltk import word_tokenize
from nltk.util import ngrams

import matplotlib.pyplot as plt
# import seaborn as sns

tqdm.pandas()
pd.set_option('max_colwidth', 100)

# from ast import literal_evals

__all__ = ['clean_text', 'strip_str_datetime']

from datetime import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta


def strip_str_datetime(date_str1, date_str2):
    if date_str1=="None" or date_str2=="None":
        return None
    return (parser.parse(date_str1) - parser.parse(date_str2)).days
#     return (datetime.strptime(date_str1, '%Y-%M-%d') -
#                          datetime.strptime(date_str2, '%Y-%M-%d')).days

def clean_text(text):
    
    text = text.replace(":\n", ": ") #internal punctuation
    text = text.replace(": \n", ": ")
    text = text.replace(",\n", ", ")
    text = text.replace(", \n", ", ")
    text = text.replace(";\n", "; ")
    text = text.replace("; \n", "; ")
    text = text.replace("'\n",  "' ")
    text = text.replace("' \n", "' ")
    text = text.replace("\"\n",  "\" ")
    text = text.replace("\" \n", "\" "): \n", ": ")


    text = re.sub(r"[^\S\n]+", " ", text)
    
    text = text.replace("==", "")
    text = text.replace("_", "")
    text = text.replace("--", "")
    text = text.replace("**", "")
    text = text.replace("\n\n", "\n")
    text = text.replace("\n \n", '\n')
        

    repl = re.findall(r"\n[a-z]", text)
    for r in repl:
        text = text.replace(r, ' '+r[-1])   
        
    repl = re.findall(r"\n [a-z]", text)
    for r in repl:
        text = text.replace(r, ' '+r[-1])   

    
    repl = re.findall(r"\n[A-Z]", text)
    for r in repl:
        text = text.replace(r, '. '+r[-1])  
        
    repl = re.findall(r"\n [A-Z]", text)
    for r in repl:
        text = text.replace(r, '. '+r[-1])  
        
        
    text = text.replace("\n", ". ")
    
    text = re.sub(r"[^\S]+", " ", text)
    
    text = re.sub(r"[.]+", ".", text)
    
    return text.lower()
