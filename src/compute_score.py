import json
import numpy as np
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import numpy as np
import torch
import copy
import json
import os
import io
import pickle
import math
import time
from tqdm import tqdm
from rouge import Rouge
import transformers
from torch import nn
import random

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

data_file = "/data/generated_msc.json"

#load dataset
list_data_dict = jload("/data/test_msc.json")
text_list = []
for test_data in list_data_dict.values():
    text_sublist = []
    for example in test_data:
        text = example['text']
        text_sublist.append(text)
    text_list.append(text_sublist)
    
targets = [sublist[-1] for sublist in text_list]

def compute_bleu(labels, preds, weights=None):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    weights = weights or (0.25, 0.25, 0.25, 0.25)
    return np.mean([sentence_bleu(references=[label],
                                  hypothesis=pred,
                                  smoothing_function=SmoothingFunction().method1,
                                  weights=weights) for label, pred in zip(labels, preds)])

def compute_bleu1(labels, preds, weights=None):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    weights = weights or (1, 0, 0, 0)
    return np.mean([sentence_bleu(references=[label],
                                  hypothesis=pred,
                                  smoothing_function=SmoothingFunction().method1,
                                  weights=weights) for label, pred in zip(labels, preds)])

def compute_bleu2(labels, preds, weights=None):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    weights = weights or (0, 1, 0, 0)
    return np.mean([sentence_bleu(references=[label],
                                  hypothesis=pred,
                                  smoothing_function=SmoothingFunction().method1,
                                  weights=weights) for label, pred in zip(labels, preds)])

def compute_distinct(sents):
    word_lst = []
    total_sent = ' '.join(sents)
    total_sent = total_sent.split(' ')
    len_total_sent = len(total_sent)
    for i in range(len_total_sent):
        word_lst.append(total_sent[i])
    gram_1_lst = list(set(word_lst))
    word_2_lst = []
    word_3_lst = []
    for sent in sents:
        sent = sent.split(' ')
        len_sent = len(sent)
        for i in range(len_sent-1):
            word_2_lst.append(sent[i] + ' ' + sent[i+1])
    for sent in sents:
        sent = sent.split(' ')
        len_sent = len(sent)
        for i in range(len_sent-2):
            word_3_lst.append(sent[i] + ' ' + sent[i+1] + ' ' + sent[i+2])
    gram_2_lst = list(set(word_2_lst))
    gram_3_lst = list(set(word_3_lst))
    dis1 = round(len(gram_1_lst) / len(word_lst), 4)
    dis2 = round(len(gram_2_lst) / len(word_2_lst), 4)
    dis3 = round(len(gram_3_lst) / len(word_3_lst), 4)
    return {'distinct1': dis1,
            'distinct2': dis2,
            'distinct3': dis3}

def compute_rouge(answers, targets):
    rouger = Rouge()
    rouge_1_f_scores = []
    rouge_2_f_scores = []
    rouge_l_f_scores = []

    for idx, (answer, target) in enumerate(zip(answers, targets)):
        try:
            scores = rouger.get_scores(answer, target)[0]  
            rouge_1_f_scores.append(scores['rouge-1']['f'])
            rouge_2_f_scores.append(scores['rouge-2']['f'])
            rouge_l_f_scores.append(scores['rouge-l']['f'])
        except ValueError as e:
            print(f"Error at index {idx}: {e}")
            print(f"Answer: {answer}")
            print(f"Target: {target}")
            continue

    avg_rouge_1_f = sum(rouge_1_f_scores) / len(rouge_1_f_scores)
    avg_rouge_2_f = sum(rouge_2_f_scores) / len(rouge_2_f_scores)
    avg_rouge_l_f = sum(rouge_l_f_scores) / len(rouge_l_f_scores)

    return {'rouge_1': avg_rouge_1_f,
            'rouge_2': avg_rouge_2_f,
            'rouge_l': avg_rouge_l_f}



with open(data_file, 'r') as f:
    answers = json.load(f)

answers = [sentence.split('<')[0].strip() for sentence in answers]

bleu = compute_bleu(answers, targets)
bleu1 = compute_bleu1(answers, targets)
bleu2 = compute_bleu2(answers, targets)
distinct = compute_distinct(answers)
rouge = compute_rouge(answers, targets)

print(bleu)
print(bleu1)
print(bleu2)
print(distinct)
print(rouge)