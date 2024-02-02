# AAPD

Project for Lead Researcher position @iris.ai

#### Time spent
31/01 14-16h: initial task setup - EDA - first running predictor


## Research article classification

### Task

open-set classification / multi-label?


### Data

Arxiv dataset: (HF_datasets)[https://huggingface.co/datasets/arxiv_dataset/blob/main/arxiv_dataset.py]

#### EDA


### Model



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