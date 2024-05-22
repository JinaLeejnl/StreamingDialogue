# StreamingDialogue: Prolonged Dialogue Learning via Long Context Compression with Minimal Losses

This repository is the official implementation of StreamingDialogue: Prolonged Dialogue Learning via Long Context Compression with Minimal Losses. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

### SMR&LMR

```train
cd src
./train.sh
./merge.sh
```

### Supervised Learning

Comment out the content between "### SMR&LMR dataset ###" and "### SMR&LMR end ###" in src\llmtuner\train\sft\workflow.py, and uncomment the content between "### Supervised learning dataset ###" and "### Supervised learning end ###".

```train
cd src
./train.sh
./merge.sh
```

## Evaluation

### Generate

```eval
cd src
python generate.py
```

### Inference

```eval
cd src
python infer.py
```

### Evaluate
>evaluate PPL
```eval
cd src
python eval_ppl.py --base_model /model/strdialogue-merge --eval_path /data/test_msc.json
```

>evaluate various metrics
```eval
cd src
python compute_score.py
```

## Pre-trained Models

You can download pretrained models here:

- [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
