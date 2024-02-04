# AAPD

Project for Lead Researcher position @iris.ai

#### Time spent

31/01 14-16h: initial task setup - EDA

02/02 20-22h: EDA + preprocessing [SYNC]

04/02 08-11h: baseline_multilabel setup - implementation testing
    17-19u: implementation tested - status: working on baseline_multilabel [SYNC]
    20-21u: implementation of zero/few-shot classification and generative models for instruction tuning (to be tested) [SYNC]


## Research article classification

### Task

(extreme) multi-label / hierarchical / multi-level learning

### Papers

- [A Survey on Extreme Multi-label Learning](link)

### Data

Arxiv dataset: (HF_datasets)[https://huggingface.co/datasets/arxiv_dataset/blob/main/arxiv_dataset.py]

#### EDA



### Experiments

Baseline: BERT
Better encoder: DeBERTa
Better pre-training: SciBERT

#### Metrics

- accuracy
- precision
- recall
- F1
- Hamming loss

## Results/Report

See ...



### Reproduction

## Installation

The scripts require [python >= 3.8](https://www.python.org/downloads/release/python-380/) to run.
We will create a fresh [virtualenvironment](https://virtualenvwrapper.readthedocs.io/en/latest/install.html) in which to install all required packages.
```sh
mkvirtualenv -p /usr/bin/python3 AAPD
```

Using poetry and the readily defined pyproject.toml, we will install all required packages
```sh
workon AAPD 
pip3 install poetry
poetry install
```

## Organization




#### ongoing notes

1. try HF loader first, then bother with installation and EDA
2. Go for baseline first fast, then try to improve