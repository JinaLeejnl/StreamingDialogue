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

torch.set_printoptions(threshold=float('inf'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

model_name = "/model/Llama-2-7b-hf" 

config = AutoConfig.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
tokenizer.pad_token_id = 0  
tokenizer.bos_token_id = 1 
tokenizer.eos_token_id = 2
tokenizer.truncation_side = "left"
tokenizer.padding_side = "right"
IGNORE_INDEX = -100
max_new_tokens = 60

model.eval()
model.cuda()

#load dataset
list_data_dict = jload("/data/personaChat_valid.json")
text_list = []
for test_data in list_data_dict.values():
    text_sublist = []
    for example in test_data:
        text = example['text']
        text_sublist.append(text)
    prompt = ' '.join(text_sublist)
    text_list.append(prompt)

file_path = "/data/generated_personachat.json"
text_list = json.dumps(text_list)
with open(file_path, "w") as file:
    file.write(text_list)


targets = [sublist[-1] for sublist in text_list]

tokenized_list = [
    [
        tokenizer(
            text,
            padding=False, 
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]
        for text in inner_list
    ]
    for inner_list in text_list
]

print("len(tokenized_list) = ", len(tokenized_list))

examples_inputs = []
for i in range(len(tokenized_list)):
    examples = [tokenizer.bos_token_id]
    for j in range(len(tokenized_list[i])-1):
        examples += tokenized_list[i][j] + [tokenizer.eos_token_id]

    examples_inputs.append(examples)

answers = []

with torch.no_grad():
    for idx, example in tqdm(enumerate(examples_inputs), total=len(examples_inputs)):
        encode_input = torch.tensor(example).reshape(1,-1)
        prompt_length = encode_input.shape[-1]
        outputs = model.generate(encode_input.cuda(), max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        answer = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=False)
        answers.append(copy.deepcopy(answer))

json_data = json.dumps(answers)

with open(file_path, "w") as file:
    file.write(json_data)