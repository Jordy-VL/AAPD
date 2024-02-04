import os
from dataclasses import dataclass, field
import random
import torch
import numpy as np


@dataclass
class CustomArguments:
    experiment_name: str = field(
        metadata={"help": "Name of the experiment"},
    )
    model_name_or_path: str = field(
        default="bert-base-uncased",  # "microsoft/deberta-v3-small", "scibert-scivocab-uncased"
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    sentence_transformer: str = field(
        default="sentence-transformers/paraphrase-mpnet-base-v2",  # "jordyvl/scibert_scivocab_uncased_sentence_transformer"
        metadata={"help": "Path to pretrained sentence transformer model"},
    )
    LLM: str = field(
        default="NousResearch/Llama-2-7b-hf",
        metadata={"help": "Path to pretrained sentence transformer model"},
    )

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def preprocess_function(example, class2id, tokenizer, label_name="cats"):
    text = example["abstract"]
    all_labels = example[label_name]
    labels = np.zeros(len(class2id))
    for label in all_labels:
        label_id = class2id[label]
        labels[label_id] = 1.0

    example = tokenizer(text, truncation=True)
    example["labels"] = labels
    return example
