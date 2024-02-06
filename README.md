# AAPD

Project for Lead Researcher position @iris.ai

#### Time spent

Total working time spent: 15h

31/01 14-16h: initial task setup - EDA

02/02 20-22h: EDA + preprocessing [SYNC]

04/02 08-11h: baseline_multilabel setup - implementation testing

    17-19u: implementation tested - status: working on baseline_multilabel [SYNC]

    20-21u: implementation of zero/few-shot classification and generative models for instruction tuning (to be tested) [SYNC]

05/02 13-15h: patch for stringlabels in HF datasets [SYNC]

    22-23h: testing zero/few-shot classification and generative models for instruction tuning  [SYNC] 

06/02 12-14h: implemented TwoWayLoss with custom HFTrainer and tested - will start final run with SciBERT (current best settings: 2e-5 lr, 64BS, 25K steps, requires >46GB GPU VRAM) [SYNC]

## Research article classification

### Task

(extreme) multi-label / hierarchical / multi-level learning

### Papers

- (DeBERTa)[https://openreview.net/forum?id=XPZIaotutsD] and (v3)[https://arxiv.org/abs/2111.09543]
- (SciBERT)[https://aclanthology.org/D19-1371/]
- (Two-way Multi-Label Loss)[https://openaccess.thecvf.com/content/CVPR2023/papers/Kobayashi_Two-Way_Multi-Label_Loss_CVPR_2023_paper.pdf]

### Data

Arxiv dataset: (HF_datasets)[https://huggingface.co/datasets/arxiv_dataset/blob/main/arxiv_dataset.py]

#### EDA

`EDA.ipynb`: hyperannotated with instructions

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

First baseline: BERT 10K train steps@6
Run summary:
                 eval/accuracy 0.9906
                       eval/f1 0.09704
                  eval/hamming 0.0094
                     eval/loss 0.03556
                eval/precision 0.78273
                   eval/recall 0.05173
                  eval/runtime 153.5857
       eval/samples_per_second 43.904
         eval/steps_per_second 7.318
                 test/accuracy 0.99066
                       test/f1 0.10248
                  test/hamming 0.00934
                     test/loss 0.03568
                test/precision 0.78642
                   test/recall 0.05481
                  test/runtime 227.935
       test/samples_per_second 43.872
         test/steps_per_second 7.313
                   train/epoch 0.03
             train/global_step 10000
           train/learning_rate 0.0
                    train/loss 0.0665
              train/total_flos 1.0442950330826088e+16
              train/train_loss 0.06653
           train/train_runtime 6679.0776
train/train_samples_per_second 8.983
  train/train_steps_per_second 1.497

Second baseline: BERT 50K train steps@6

wandb:                  test/accuracy 0.99346
wandb:                        test/f1 0.57809
wandb:                   test/hamming 0.00654
wandb:                      test/loss 0.0195
wandb:                 test/precision 0.77619
wandb:                    test/recall 0.46054

See wandb report @ ...



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