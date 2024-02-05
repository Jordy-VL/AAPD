#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2024 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "3.0"

## Necessary installs
#!pip install datasets transformers evaluate sentencepiece accelerate

import os
import numpy as np
from argparse import Namespace
from datasets import load_from_disk, load_dataset, ClassLabel, Sequence
from transformers import AutoTokenizer, HfArgumentParser, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import wandb
import evaluate
from myutils import CustomArguments, seed_everything, preprocess_function


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
        tags=["multi-label"],
        config={k: v for k, v in args.__dict__.items() if v is not None},
    )

    ## Load the dataset and initialize the classes
    # DATAROOT = os.path.join(os.path.dirname(__file__), "..", "data")
    # dataset = load_from_disk(os.path.join(DATAROOT, "arxiv_dataset_prepped"))

    # # make another subset of validation - still too large
    # simple_validation = dataset["validation"].train_test_split(
    #     test_size=0.1, seed=args.seed, stratify_by_column="strlabel"
    # )
    # dataset["simple_validation"] = simple_validation["test"]

    dataset = load_dataset("jordyvl/arxiv_dataset_prep")  # new version for continued comparisons

    label_name = "cats"

    classes = sorted(set([c for cats in dataset["train"][label_name] for c in cats]))
    dataset.cast_column(label_name, Sequence(ClassLabel(names=classes)))  # .class_encode_column("cats")
    class2id = {class_: id for id, class_ in enumerate(classes)}
    id2class = {id: class_ for class_, id in class2id.items()}

    print(f"Classes: {len(classes)}")
    print(f"Class2id: {class2id}")
    print(f"Id2class: {id2class}")

    ## Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenized_dataset = dataset.map(lambda example: preprocess_function(example, class2id, tokenizer))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(classes),
        id2label=id2class,
        label2id=class2id,
        problem_type="multi_label_classification",
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["simple_validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    try:
        train_results = trainer.train()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
    except KeyboardInterrupt as e:
        print(e)

    subsample_test = tokenized_dataset["test"].select(list(range(0, 10000)))  # takes 30 minutes on desktop
    trainer.evaluate(eval_dataset=subsample_test, metric_key_prefix="test")  # 20K samples is enough?

    # print some example outputs
    trainer.push_to_hub(f"Saving best model of {args.experiment_name} to hub")
    print("Example outputs to check:")
    subset = tokenized_dataset["test"].select(list(range(0, 100)))
    print(trainer.predict(subset), "vs. GT: ", subset["labels"])  # to be logged for debugging purposes


if __name__ == "__main__":
    main()
