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
- [X] reproduction test and scripts
- [X] changed imports to conda environment yml
- [WONTDO] report ANLS in wandb (might be a single run given time to be spent)
- [WONTDO] port metrics mAP (classwise) and mAP (samplewise) from (repo)[https://github.com/tk1980/TwoWayMultiLabelLoss/blob/main/utils/utils.py]


## Results/Report

See wandb report @ https://wandb.ai/jordy-vlan/scientific-text-classification

### Reproduction

To reproduce the results of the report, one can run the commands of the models in wandb.
For example, to reproduce the results of the ((current) best reported model)[https://wandb.ai/jordy-vlan/scientific-text-classification/runs/zk81z3qc/overview?workspace=user-jordy-vlan], one can run the following command:

```bash
python src/baseline_multilabel.py --experiment_name SciBERT_twowayloss_25K_bs64 --model_name_or_path allenai/scibert_scivocab_uncased --output_dir ../results --seed 42 --evaluation_strategy steps --per_device_train_batch_size 64 --gradient_accumulation_steps 1 --learning_rate 2e-5 --num_train_epochs 1 --max_steps 25000 --logging_strategy steps --logging_steps 0.05 --save_steps 0.2 --eval_steps 0.2 --criterion TwoWayLoss --Tp 4.0 --Tn 1.0
```

#### Installation

The scripts require [python >= 3.8](https://www.python.org/downloads/release/python-380/) to run and a conda environment with the following packages:

```bash
    conda env create -f environment.yml  # creates the environment
    conda activate aapd  # activates the environment
```

## Organization

#### Time spent

Total working time spent: 17h

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

07/02 

    14-16h: fixed LLM evaluation - SetFit training slowly, issues in evaluation with SetFitTrainer
