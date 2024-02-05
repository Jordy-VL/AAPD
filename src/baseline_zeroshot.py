#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2024 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "3.0"

#!pip3 install sentence_transformers setfit

import os
import numpy as np
from argparse import Namespace
from datasets import load_dataset, load_from_disk, ClassLabel, Sequence
from transformers import HfArgumentParser
from transformers import TrainingArguments
import wandb
import evaluate
from myutils import CustomArguments, seed_everything

from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, Trainer


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
    parser = HfArgumentParser((CustomArguments, TrainingArguments))
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

    ## Load the dataset and initialize the classes; https://github.com/huggingface/setfit/issues/226
    dataset = load_dataset("jordyvl/arxiv_dataset_prep")
    from pdb import set_trace

    set_trace()
    classes = sorted(set([c.strip() for cats in dataset["train"]["strlabel"] for c in cats.split(";")]))
    dataset.cast_column("strlabel", Sequence(ClassLabel(names=classes)))  # .class_encode_column("cats")
    class2id = {class_: id for id, class_ in enumerate(classes)}
    id2class = {id: class_ for class_, id in class2id.items()}
    # for setfit compatibility
    dataset.rename_column("strlabel", "labels")
    dataset.rename_column("abstract", "text")

    print(f"Classes: {len(classes)}")
    print(f"Class2id: {class2id}")
    print(f"Id2class: {id2class}")

    #'jordyvl/scibert_scivocab_uncased_sentence_transformer'
    model = SetFitModel.from_pretrained(args.sentence_transformer, multi_target_strategy="one-vs-rest")

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.experiment_name),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,  # override for now
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        weight_decay=0.01,  # override default
        warmup_ratio=0.1,  # override default
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=3,
        push_to_hub=True,
        hub_strategy="end",
        load_best_model_at_end=True,
        run_name=args.experiment_name,
        hub_model_id=args.experiment_name,
        label_smoothing_factor=args.label_smoothing_factor,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        loss_class=CosineSimilarityLoss,
        batch_size=args.per_device_train_batch_size,
        num_epochs=args.num_train_epochs,  # Number of epochs to use for contrastive learning
        metric="f1",
        metric_kwargs={"average": "micro"},
        num_iterations=5,  # Number of text pairs to generate for contrastive learning
    )

    try:
        train_results = trainer.train()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
    except KeyboardInterrupt as e:
        print(e)

    zeroshot_metrics = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
    trainer.push_to_hub(f"Saving best model of {args.experiment_name} to hub")


if __name__ == "__main__":
    main()
