#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2024 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "3.0"

#!pip3 install sentence_transformers setfit --upgrade

import os
import numpy as np
from argparse import Namespace
from datasets import load_dataset, ClassLabel, Sequence
from transformers import HfArgumentParser
from transformers import TrainingArguments as HFTrainingArguments
import wandb
import evaluate
from myutils import CustomArguments, seed_everything, preprocess_zero_function

from setfit import SetFitModel, Trainer, TrainingArguments


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hamming_loss(y_true, y_pred):
    return np.mean(y_true != y_pred)


clf_metrics = evaluate.combine(["accuracy", "precision", "recall", "f1"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    references = labels.astype(int).reshape(-1)
    batch_metrics = clf_metrics.compute(predictions=predictions, references=references)
    batch_metrics["hamming"] = hamming_loss(references, predictions)
    return batch_metrics


def main():
    parser = HfArgumentParser((CustomArguments, HFTrainingArguments))
    custom_args, prior_training_args = parser.parse_args_into_dataclasses()
    for k, v in custom_args.__dict__.items():
        print(k, v)
    args = Namespace(**vars(custom_args), **vars(prior_training_args))
    seed_everything(args.seed)

    wandb.init(
        project="scientific-text-classification",
        name=args.experiment_name if args.experiment_name else None,
        tags=["zero-shot"],
        config={k: v for k, v in args.__dict__.items() if v is not None},
    )
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.experiment_name),
        max_steps=args.max_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        batch_size=args.per_device_train_batch_size,
        save_total_limit=3,
        load_best_model_at_end=True,
        run_name=args.experiment_name,
        num_iterations=20,
    )

    ## Load the dataset and initialize the classes; https://github.com/huggingface/setfit/issues/226
    dataset = load_dataset("jordyvl/arxiv_dataset_prep")
    classes = sorted(set([c.strip() for cats in dataset["train"]["strlabel"] for c in cats.split(";")]))
    class2id = {class_: id for id, class_ in enumerate(classes)}

    print(f"Classes: {len(classes)}")

    # for setfit compatibility need 'label' (onehot) and 'text' columns
    dataset = dataset.map(lambda x: preprocess_zero_function(x, class2id), remove_columns=["strlabel"]).rename_column(
        "abstract", "text"
    )

    # DEV: https://github.com/huggingface/setfit/issues/472 cannot use whole dataset due to memory issues; raised ticket
    ## numpy.core._exceptions._ArrayMemoryError: Unable to allocate 3.72 TiB for an array with shape (2022879, 2022879) and data type bool
    ## related to https://stackoverflow.com/questions/40617199/generate-n-choose-2-combinations-in-python-on-very-large-data-sets
    ## backoff solution: subsample training data to 5% of the original size
    dataset["train"] = dataset["train"].select(
        list(range(0, int(0.05 * len(dataset["train"]))))
    )  # permuted by default from train test split

    model = SetFitModel.from_pretrained(
        args.sentence_transformer,
        multi_target_strategy=args.multi_target_strategy,
        use_differentiable_head=args.use_differentiable_head,
        labels=classes,
    )

    # Create trainer
    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["simple_validation"],
        metric="f1",  # DEV: no evaluate.compute metrics availability, TODO: custom compute_metrics and evaluation loop
        metric_kwargs={"average": "micro"},
    )

    try:
        train_results = trainer.train()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
    except KeyboardInterrupt as e:
        print(e)

    zeroshot_metrics = trainer.evaluate(
        eval_dataset=dataset["test"].select(list(range(0, 10000))), metric_key_prefix="test"
    )
    trainer.push_to_hub(f"Saving best model of {args.experiment_name} to hub")


if __name__ == "__main__":
    main()
