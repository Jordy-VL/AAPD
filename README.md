# AAPD

Project for Lead Researcher position @iris.ai

## Research article classification

Report under `report/main.pdf`

All code under `src`

### Task

(extreme) multi-label / hierarchical / multi-level text classification

### Papers

- [DeBERTa](https://openreview.net/forum?id=XPZIaotutsD) and [v3](https://arxiv.org/abs/2111.09543)
- [SciBERT](https://aclanthology.org/D19-1371/)
- [Two-way Multi-Label Loss](https://openaccess.thecvf.com/content/CVPR2023/papers/Kobayashi_Two-Way_Multi-Label_Loss_CVPR_2023_paper.pdf)

### Data

Arxiv dataset: (HF_datasets)[https://huggingface.co/datasets/arxiv_dataset/blob/main/arxiv_dataset.py]

#### EDA

`EDA.ipynb`: hyperannotated with commentary

### Experiments

Baseline: BERT
- Better encoder: DeBERTa
- Better pre-training: SciBERT
- Better loss: Two-way Multi-Label Loss

Alternate approaches:
- Few-shot classification (Setfit)
- Generative models for instruction tuning (Llama2)

#### Metrics

- accuracy
- precision
- recall
- F1
- Hamming loss

TODO:
- [ ] report ANLS in wandb (might be a single run given time to be spent)
- [ ] port metrics mAP (classwise) and mAP (samplewise) from (repo)[https://github.com/tk1980/TwoWayMultiLabelLoss/blob/main/utils/utils.py]
- [ ] reproduction test and scripts
- [ ] imports in pyproject.toml


## Results/Report

See wandb report @ https://wandb.ai/jordy-vlan/scientific-text-classification

### Reproduction


#### Installation

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


reproduce.sh: script to reproduce the final run

```sh
bash reproduce.sh
```

## Organization

#### Time spent

Total working time spent: 15h

31/01 14-16h: initial task setup - EDA

02/02 20-22h: EDA + preprocessing [SYNC]

04/02 
    
    08-11h: baseline_multilabel setup - implementation testing

    17-19u: implementation tested - status: working on baseline_multilabel [SYNC]

    20-21u: implementation of zero/few-shot classification and generative models for instruction tuning (to be tested) [SYNC]

05/02 
    
    13-15h: patch for stringlabels in HF datasets [SYNC]

    22-23h: testing zero/few-shot classification and generative models for instruction tuning  [SYNC] 

06/02 
    
    12-14h: implemented TwoWayLoss with custom HFTrainer and tested - will start final run with SciBERT (current best settings: 2e-5 lr, 64BS, 25K steps, requires >46GB GPU VRAM) [SYNC]

    14-15h: boilerplated report

    19-20h: finalizing report - to fill in results


