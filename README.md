# StreamingDialogue: Prolonged Dialogue Learning via Long Context Compression with Minimal Losses

This repository is the official implementation of StreamingDialogue: Prolonged Dialogue Learning via Long Context Compression with Minimal Losses. The code is developed based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

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

1. Modify the file `/src/llmtuner/train/sft/workflow.py`:
   - Comment out lines 157-240.
   - Uncomment lines 83-94.

2. Modify the file `/model/llama/modeling_llama.py`:
   - Comment out lines 1049-1052.
   - Uncomment lines 1029-1032.

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
