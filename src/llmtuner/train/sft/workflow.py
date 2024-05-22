# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, Optional, List
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from datasets import load_dataset
from datasets import Dataset, IterableDataset

from llmtuner.data import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import get_logits_processor
from llmtuner.extras.ploting import plot_loss
from llmtuner.model import load_model_and_tokenizer
from llmtuner.train.sft.metric import ComputeMetrics
from llmtuner.train.sft.trainer import CustomSeq2SeqTrainer
from llmtuner.train.utils import create_modelcard_and_push
import copy
import os
import torch
import json
import io
import random
import numpy as np
from itertools import chain

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments

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

def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):

    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")
    tokenizer.pad_token_id = 0  
    tokenizer.bos_token_id = 1 
    tokenizer.eos_token_id = 2
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"
    IGNORE_INDEX = -100

    ################################################################# msc dataset ##########################################
    list_data_dict = jload("/data/train_msc.json")
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

    print("len(tokenized_list) = ", len(tokenized_list))

    # ### Supervised learning dataset ###
    # examples_inputs = []
    # for i in range(len(tokenized_list)):
    #     examples = [tokenizer.bos_token_id]
    #     for j in range(len(tokenized_list[i])):
    #         examples += tokenized_list[i][j] + [tokenizer.eos_token_id]
        
    #     examples_inputs.append(examples)

    # labels = copy.deepcopy(examples_inputs)
    # slflags = [0] * len(labels)
    # ### Supervised learning end ###


    # ################### Build LMR dataset ###################
    # random_seed=42
    # random.seed(random_seed)

    # whole_inputs = []
    # whole_labels = []
    # for i in range(len(tokenized_list)):
    #     for j in range(1, len(tokenized_list[i])):
    #         whole_inputs.append(tokenized_list[i][j - 1] + [tokenizer.eos_token_id] + tokenized_list[i][j] + [tokenizer.eos_token_id])
    #         whole_labels.append(tokenized_list[i][j - 1] + [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(tokenized_list[i][j]) + 1))
    
    # turns = 24
    # examples_inputs = []
    # labels = []
    # sample_len = 8000
    # for i in range(sample_len):
    #     sampled_indices = random.sample(range(len(whole_inputs)), turns)
    #     sampled_inputs = [whole_inputs[i] for i in sampled_indices]
    #     sampled_labels = [whole_labels[i] for i in sampled_indices]

    #     random_index = random.randint(0, len(sampled_inputs) - 1)
    #     selected_input = sampled_inputs[random_index]
    #     selected_label = sampled_labels[random_index]

    #     examples = [tokenizer.bos_token_id] + list(chain.from_iterable(sampled_inputs))
    #     label = copy.deepcopy(examples)
    #     examples += selected_input
    #     label += selected_label

    #     examples_inputs.append(examples)
    #     labels.append(label)

    # slflags = [1] * len(labels)



    # ################### Build SMR dataset ###################
    # random_seed=42
    # random.seed(random_seed)
    # turns = 24
    # sample_len = 8000
    # flattened_list = [item for sublist in tokenized_list for item in sublist]
    # print("flattened_list len:", len(flattened_list))
    # examples_inputs = []
    # labels = []
    # for i in range(sample_len):
    #     random_inputs = random.sample(flattened_list, turns)
    #     examples = [tokenizer.bos_token_id]
    #     label = [IGNORE_INDEX]
    #     for j in range(turns):
    #         examples += random_inputs[j] + [tokenizer.eos_token_id] + random_inputs[j] + [tokenizer.eos_token_id]
    #         label += [IGNORE_INDEX] * (len(random_inputs[j]) + 1) + random_inputs[j] + [tokenizer.eos_token_id]
        
    #     examples_inputs.append(examples)
    #     labels.append(label)
    
    # slflags = [0] * len(labels)



    ### SMR&LMR dataset ###
    random_seed=42
    random.seed(random_seed)
    turns = 24
    sample_len = 8000

    whole_inputs = []
    whole_labels = []
    for i in range(len(tokenized_list)):
        for j in range(1, len(tokenized_list[i])):
            whole_inputs.append(tokenized_list[i][j - 1] + [tokenizer.eos_token_id] + tokenized_list[i][j] + [tokenizer.eos_token_id])
            whole_labels.append(tokenized_list[i][j - 1] + [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(tokenized_list[i][j]) + 1))
    
    all_long_inputs = []
    all_long_labels = []
    for i in range(sample_len):
        sampled_indices = random.sample(range(len(whole_inputs)), turns)
        sampled_inputs = [whole_inputs[i] for i in sampled_indices]
        sampled_labels = [whole_labels[i] for i in sampled_indices]
        all_long_inputs.extend(sampled_inputs)
        all_long_labels.extend(sampled_labels)

    actuall_turns = 24
    actuall_sample_len = sample_len * turns // actuall_turns
    long_inputs = []
    long_labels = []
    for i in range(actuall_sample_len):
        sampled_inputs = all_long_inputs[i * actuall_turns: (i + 1) * actuall_turns]
        sampled_labels = all_long_labels[i * actuall_turns: (i + 1) * actuall_turns]
        random_index = random.randint(0, len(sampled_inputs) - 1)
        selected_input = sampled_inputs[random_index]
        selected_label = sampled_labels[random_index]

        examples = [tokenizer.bos_token_id] + list(chain.from_iterable(sampled_inputs))
        label = copy.deepcopy(examples)
        examples += selected_input
        label += selected_label

        long_inputs.append(examples)
        long_labels.append(label)

    ###
    turns = 24
    sample_len = 48000
    flattened_list = [item for sublist in tokenized_list for item in sublist]
    all_short_inputs = []
    for i in range(sample_len):
        random_inputs = random.sample(flattened_list, turns)
        all_short_inputs.extend(random_inputs)
    
    actuall_turns = 28
    actuall_sample_len = sample_len * turns // actuall_turns
    short_inputs = []
    short_labels = []
    for i in range(actuall_sample_len):
        random_inputs = all_short_inputs[i * actuall_turns: (i + 1) * actuall_turns]
        examples = [tokenizer.bos_token_id]
        label = [IGNORE_INDEX]
        for j in range(actuall_turns):
            examples += random_inputs[j] + [tokenizer.eos_token_id] + random_inputs[j] + [tokenizer.eos_token_id]
            label += [IGNORE_INDEX] * (len(random_inputs[j]) + 1) + random_inputs[j] + [tokenizer.eos_token_id]
        
        short_inputs.append(examples)
        short_labels.append(label)
    
    # shuffle
    short_data = list(zip(short_inputs, short_labels))
    long_data = list(zip(long_inputs, long_labels))

    all_data = short_data + long_data

    flags = [0] * len(short_data) + [1] * len(long_data)

    combined_data = list(zip(all_data, flags))
    random.shuffle(combined_data)

    shuffled_data, shuffled_flags = zip(*combined_data)

    shuffled_inputs, shuffled_labels = zip(*shuffled_data)

    examples_inputs = list(shuffled_inputs)
    labels = list(shuffled_labels)
    slflags = list(shuffled_flags)
    ### SMR&LMR end ###




    model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "flags": []}

    for example, label, flag in zip(examples_inputs, labels, slflags):
        model_inputs["input_ids"].append(example)
        model_inputs["attention_mask"].append([1] * len(example))
        model_inputs["labels"].append(label)
        model_inputs["flags"].append(flag)

    dataset = Dataset.from_dict(model_inputs)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left" # use left-padding in generation

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=1 if tokenizer.padding_side == "right" else None, # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    training_args.remove_unused_columns=False

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **split_dataset(dataset, data_args, training_args)
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate: # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate: # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)