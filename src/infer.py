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
import subprocess
import torch

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

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

model_name = "/model/strdialogue-merge"

config = AutoConfig.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
tokenizer.pad_token_id = 0  
tokenizer.bos_token_id = 1 
tokenizer.eos_token_id = 2
tokenizer.truncation_side = "left"
tokenizer.padding_side = "right"
IGNORE_INDEX = -100
max_new_tokens = 32

model.eval()
model.cuda()

#load dataset
list_data_dict = jload("/data/test_msc.json")
text_list = []
for test_data in list_data_dict.values():
    text_sublist = []
    for example in test_data:
        text = example['text']
        text_sublist.append(text)
    text_list.append(text_sublist)

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

examples_inputs = []
for i in range(len(tokenized_list)):
    examples = [tokenizer.bos_token_id]
    for j in range(len(tokenized_list[i]) - 1):
        examples += tokenized_list[i][j] + [tokenizer.eos_token_id]

    examples_inputs.append(examples)


with torch.no_grad():
    for idx, example in tqdm(enumerate(examples_inputs), total=len(examples_inputs)):
        encode_input = torch.tensor(example).reshape(1,-1)
        prompt_length = encode_input.shape[-1]
        indices = torch.where(encode_input[0] == 2)[0]
        indices = torch.cat([torch.tensor([0]), indices])
        indices = torch.cat([indices[:-2], torch.arange(indices[-2], prompt_length)]).to(device)
        outputs = model(
            input_ids=encode_input.cuda(),
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        past_key_values = [
            [
                k[:, :, indices, ...],
                v[:, :, indices, ...],
            ]
            for k, v in past_key_values
        ]

        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids = [pred_token_idx.item()]
        pos = 0
        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids.append(pred_token_idx.item())
            generated_text = (
                tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=False,
                )
                .strip()
                .split(" ")
            )

            now = len(generated_text) - 1
            if now > pos:
                print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                pos = now

            if pred_token_idx == tokenizer.eos_token_id:
                break

        print(" ".join(generated_text[pos:]), flush=True)