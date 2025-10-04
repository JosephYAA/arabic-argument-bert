"""
Model Training and Evaluation Module for Arabic Argument Mining.

This module handles training and evaluation of BERT models for different
argument mining tasks using the HuggingFace Transformers library.
"""

from typing import Any

import evaluate as evaluate_metric
import numpy as np
import torch
from datasets import DatasetDict
from seqeval.metrics import classification_report as seqeval_classification_report
from sklearn.metrics import classification_report as sklearn_classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .config import (
    ARGCLASS_LABELS,
    ARGDET_SENT_LABELS,
    ARGDET_TOK_LABELS,
    END_TO_END_LABELS,
    LEARNING_RATE,
    BERT_MODEL_NAME,
    NUM_TRAIN_EPOCHS,
    OUTPUT_DIR,
    SEQUENCE_BATCH_SIZE,
    TOKEN_BATCH_SIZE,
)
from .dataset_processing import create_all_datasets


class ModelTrainer:
    """
    Handles training and evaluation of BERT models for argument mining tasks.
    """

    def __init__(self, model_name: str = BERT_MODEL_NAME):
        """
        Initialize the model trainer.

        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.seqeval = evaluate_metric.load("seqeval")  # Using renamed import
        self.f1_metric = evaluate_metric.load("f1")  # Using renamed import

    def tokenize_and_align_labels(
        self, examples: dict[str, list], label_column: str = "tags"
    ):
        """
        Tokenize inputs and align labels for token classification tasks.

        Args:
            examples: Batch of examples
            label_column: Name of the label column

        Returns:
            Tokenized inputs with aligned labels
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples[label_column]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics_token_classification(self, eval_pred):
        """
        Compute metrics for token classification tasks.

        Args:
            eval_pred: Evaluation predictions from trainer

        Returns:
            dictionary of computed metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (-100)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def compute_metrics_sequence_classification(self, eval_pred):
        """
        Compute metrics for sequence classification tasks.

        Args:
            eval_pred: Evaluation predictions from trainer

        Returns:
            dictionary of computed metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        results = self.f1_metric.compute(
            predictions=predictions, references=labels, average="macro"
        )
        return results

    def train_token_classification(
        self,
        dataset: DatasetDict,
        task_name: str,
        labels: list[str],
        output_dir: str | None = None,
    ) -> Trainer:
        """
        Train a token classification model.

        Args:
            dataset: Dataset to train on
            task_name: Name of the task
            labels: list of label names
            output_dir: Output directory for model

        Returns:
            Trained model trainer
        """
        self.label_list = labels

        # Create label mappings
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in enumerate(labels)}

        # Tokenize dataset
        tokenized_dataset = dataset.map(self.tokenize_and_align_labels, batched=True)

        # Create data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        # Load model
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )

        # Set output directory
        if output_dir is None:
            output_dir = f"{OUTPUT_DIR}/{task_name}"

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=TOKEN_BATCH_SIZE,
            per_device_eval_batch_size=TOKEN_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1,
            seed=98,
            report_to="none",
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["dev"],
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics_token_classification,
        )

        # Train model
        print(f"Training {task_name} model...")
        trainer.train()

        # Evaluate on test set
        if "test" in tokenized_dataset:
            print(f"Evaluating {task_name} model on test set...")
            test_dataset = tokenized_dataset["test"]
            predictions, labels, _ = trainer.predict(test_dataset)
            predictions = np.argmax(predictions, axis=2)

            true_predictions = [
                [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            print("SeqEval Classification Report:")
            print(seqeval_classification_report(true_labels, true_predictions, 4))

        return trainer

    def train_sequence_classification(
        self,
        dataset: DatasetDict,
        task_name: str,
        labels: list[str],
        output_dir: str | None = None,
    ) -> Trainer:
        """
        Train a sequence classification model.

        Args:
            dataset: Dataset to train on
            task_name: Name of the task
            labels: list of label names
            output_dir: Output directory for model

        Returns:
            Trained model trainer
        """
        # Create label mappings
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in enumerate(labels)}

        # Tokenize dataset
        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)

        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        # Create data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )

        # Set output directory
        if output_dir is None:
            output_dir = f"{OUTPUT_DIR}/{task_name}"

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=SEQUENCE_BATCH_SIZE,
            per_device_eval_batch_size=SEQUENCE_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1,
            seed=98,
            report_to="none",
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["dev"],
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics_sequence_classification,
        )

        # Train model
        print(f"Training {task_name} model...")
        trainer.train()

        # Evaluate on test set
        if "test" in tokenized_dataset:
            print(f"Evaluating {task_name} model on test set...")
            test_dataset = tokenized_dataset["test"]
            predictions, labels, _ = trainer.predict(test_dataset)
            predictions = np.argmax(predictions, axis=1)

            if task_name == "argument_detection_sentence":
                target_names = ARGDET_SENT_LABELS
            else:
                target_names = ARGCLASS_LABELS

            print("Sklearn Classification Report:")
            print(
                sklearn_classification_report(
                    labels, predictions, target_names=target_names, digits=4
                )
            )
        return trainer

    def train_end_to_end_model(self, dataset: DatasetDict) -> Trainer:
        """Train end-to-end argument component identification model."""
        return self.train_token_classification(dataset, "end_to_end", END_TO_END_LABELS)

    def train_argument_detection_token_model(self, dataset: DatasetDict) -> Trainer:
        """Train token-level argument detection model."""
        return self.train_token_classification(
            dataset, "argument_detection_token", ARGDET_TOK_LABELS
        )

    def train_argument_detection_sentence_model(self, dataset: DatasetDict) -> Trainer:
        """Train sentence-level argument detection model."""
        return self.train_sequence_classification(
            dataset, "argument_detection_sentence", ARGDET_SENT_LABELS
        )

    def train_argument_classification_model(self, dataset: DatasetDict) -> Trainer:
        """Train argument component classification model."""
        return self.train_sequence_classification(
            dataset, "argument_classification", ARGCLASS_LABELS
        )


def train_all_models(datasets: dict[str, DatasetDict]) -> dict[str, Trainer]:
    """
    Train models for all three tasks.

    Args:
        datasets: dictionary containing datasets for all tasks

    Returns:
        dictionary containing trained models
    """
    trainer = ModelTrainer()
    models = {}

    # Train end-to-end model
    if "end_to_end" in datasets:
        print("Training end-to-end model...")
        models["end_to_end"] = trainer.train_end_to_end_model(datasets["end_to_end"])

    # Train argument detection model
    if "argument_detection_token" in datasets:
        print("Training argument detection token model...")
        models["argument_detection_token"] = (
            trainer.train_argument_detection_token_model(
                datasets["argument_detection_token"]
            )
        )

    if "argument_detection_sentence" in datasets:
        print("Training argument detection sentence model...")
        models["argument_detection_sentence"] = (
            trainer.train_argument_detection_sentence_model(
                datasets["argument_detection_sentence"]
            )
        )

    # Train argument classification model
    if "argument_classification" in datasets:
        print("Training argument classification model...")
        models["argument_classification"] = trainer.train_argument_classification_model(
            datasets["argument_classification"]
        )

    return models


if __name__ == "__main__":
    # Example usage

    # Create datasets
    datasets = create_all_datasets("./data/annotated")

    # Train all models
    trained_models = train_all_models(datasets)

    print("All models trained successfully!")
