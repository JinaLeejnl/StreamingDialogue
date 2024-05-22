import json
import pickle
from sys import argv

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.utils import InputExample
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.nn.parallel import DistributedDataParallel
from datasets import load_dataset
import matplotlib.pyplot as plt
from contextlib import nullcontext
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
import transformers
from rouge import Rouge
from torch import nn

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



def get_dataloader(input_examples, tokenizer, device, batch_size=256):
    features = convert_examples_to_features(
        input_examples,
        tokenizer,
        label_list=['0', '1', '2'],
        max_length=128,
        output_mode='classification',
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long).to(device)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def read_data(answers, file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data_list = json.load(file)

    personas = data_list
    preds = answers

    examples = []
    cnt = 0
    for persona_list, hyp in zip(personas, preds):
        for persona in persona_list: 
            examples.append(InputExample(str(cnt), hyp, persona , '0'))
            cnt += 1
    
    print(cnt)
    
    return examples, cnt

if __name__ == '__main__':
    data_file = "/data/generated_personachat.json"
    with open(data_file, 'r') as f:
        answers = json.load(f)
    answers = [sentence.split('<')[0].strip() for sentence in answers]

    file_path = "/data/valid_other_original_personachat.json"

    output_file = './output.txt'

    tokenizer = AutoTokenizer.from_pretrained('/model/entailment_nli_model')
    model = AutoModelForSequenceClassification.from_pretrained('/model/entailment_nli_model')
    device = torch.device('cuda:1')
    model.to(device)
    model.eval()

    input_examples, num = read_data(answers, file_path)

    train_dataloader = get_dataloader(input_examples, tokenizer, device, batch_size=512)
    all_logits = None
    
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            if all_logits is None:
                all_logits = outputs[1].cpu().detach()
            else: 
                all_logits = torch.cat((all_logits, outputs[1].cpu().detach()), dim=0)

    results = torch.argmax(all_logits, dim=1)
    cnt = 0
    for i, res in enumerate(results):
        cnt = cnt + (res - 1)
    print ('consistence score is ', cnt/1000)
    with open(output_file + '.txt', 'w') as f:
        f.write(str(cnt/1000))

    all_logits = all_logits.numpy()
    with open(output_file + '.bin', 'wb') as f:
        pickle.dump(all_logits, f)
