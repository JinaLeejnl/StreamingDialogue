import os
import sys
import math
import io
import copy
import json
import torch
import argparse
import textwrap
import transformers
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import GenerationConfig, TextStreamer
from typing import List, Dict
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier

torch.set_printoptions(threshold=float('inf'))

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

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

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="/model/Llama-2-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--eval_path', type=str, default="/data/personaChat_valid.json")
    args = parser.parse_args()
    return args

def _tokenize_fn(my_list: List[List[str]], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        [
            tokenizer(
                text,
                return_tensors="pt",
                padding=False, 
                truncation=False,
                add_special_tokens=False,
            )
            for text in inner_list
        ]
        for inner_list in my_list
    ]

    input_ids = labels = [
        [tokenized.input_ids[0] for tokenized in inner_list]
        for inner_list in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
    )


def preprocess(
    text_list: List[List[str]],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""

    text__tokenized = _tokenize_fn(text_list, tokenizer)
    input_ids = text__tokenized["input_ids"]
    sentences = []
    labels_all = []
    for i in range(len(input_ids)):
        sentence = torch.tensor(tokenizer.bos_token_id).unsqueeze(0)
        label_all = torch.tensor(tokenizer.bos_token_id).unsqueeze(0)
        for j in range(len(input_ids[i])):
            token = input_ids[i][j]
            sentence = torch.cat((sentence, token, torch.tensor(tokenizer.eos_token_id).unsqueeze(0)), dim=0)
            label_all = copy.deepcopy(sentence)
            if j == 0:
                first_len = len(label_all)

        label_all[:first_len] = IGNORE_INDEX
        sentences.append(sentence)
        labels_all.append(label_all)

    return dict(input_ids=sentences, labels=labels_all)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Formatting inputs...")
        '''
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        '''

        text_list = []
        for test_data in list_data_dict.values():
            text_sublist = []
            for example in test_data:
                text = example['text']
                text_sublist.append(text)
            
            text_list.append(text_sublist)

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(text_list, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def main(args):
    device = "cuda:2"
    torch.cuda.set_device(device)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
    ).to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token_id = 0  
    tokenizer.bos_token_id = 1 
    tokenizer.eos_token_id = 2
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    model.eval()
    
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.eval_path)

    input_ids = eval_dataset.input_ids
    labels = eval_dataset.labels

    nlls_all = []

    with torch.no_grad():
        example_num = len(input_ids)
        for i in tqdm(range(example_num)):
            input_id = input_ids[i].reshape(1, -1)
            input_id = input_id.pin_memory().to(device, non_blocking=True)
            outputs = model(input_ids = input_id, attention_mask = input_id.ne(tokenizer.pad_token_id))
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            logits = logits[..., :-1, :].contiguous()
            label = labels[i].reshape(1, -1)
            label = label[..., 1:].contiguous()
            
            label = label.to(device)

            log_probs = -nn.functional.log_softmax(logits, dim=-1)
            if label.dim() == log_probs.dim() - 1:
                label = label.unsqueeze(-1)
            
            padding_mask_all = label.eq(IGNORE_INDEX)
            label = torch.clamp(label, min=0)
            nll_loss_all = log_probs.gather(dim=-1, index=label)
            nll_loss_all.masked_fill_(padding_mask_all, 0.0)
            num_active_elements_all = padding_mask_all.numel() - padding_mask_all.long().sum()
            nll_loss_all = nll_loss_all.sum() / num_active_elements_all
            nlls_all.append(nll_loss_all)
    
    ppl_all = torch.exp(torch.stack(nlls_all).mean())

    print("PPL:", ppl_all)


if __name__ == "__main__":
    args = parse_config()
    main(args)