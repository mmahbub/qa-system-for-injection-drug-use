import warnings
warnings.filterwarnings("ignore")

import re
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
    
    punc_list = [",", ":", ";", "'", "\""]
    
    text = re.sub(r"[==]+", "", text)
    text = re.sub(r"[__]+", "", text)
    text = re.sub(r"[--]+", "", text)
    text = re.sub(r"[**]+", "", text)
    text = re.sub(r"[\?]+", "?", text)
    text = re.sub(r"[\!]+", "!", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\n\s+", "\n ", text)
    text = re.sub(r"[.]+", ".", text)
    text = re.sub(r' +', " ", text)
    
    for punc in punc_list:
        text = re.sub(rf"{punc}+", punc, text)
    
    for punc in punc_list:
        text = re.sub(rf"\n[{punc}]", punc, text)
        text = re.sub(rf"\n [{punc}]", punc, text)
        text = re.sub(rf"[{punc}]\n", f"{punc} ", text)
        text = re.sub(rf"[{punc}] \n", f"{punc} ", text)
    
    text = re.sub("\n[.]", ". ", text)
    text = re.sub("\n [.]", ". ", text)
    
    repl = re.findall(r"\n[a-z]", text)
    for r in repl:
        text = re.sub(r, ' '+r[-1], text)   
        
    repl = re.findall(r"\n [a-z]", text)
    for r in repl:
        text = re.sub(r, ' '+r[-1], text)   
    
    repl = re.findall(r"\n[A-Z]", text)
    for r in repl:
        text = re.sub(r, '. '+r[-1], text)  
        
    repl = re.findall(r"\n [A-Z]", text)
    for r in repl:
        text = re.sub(r, '. '+r[-1], text)  
    
    text = re.sub("\n", ". ", text)
    text = re.sub(r"[^\S]+", " ", text)    
    text = re.sub(r"[.]+", ".", text)
    
    return text.lower()
