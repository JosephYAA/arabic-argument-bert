"""
Dataset Processing Module for Arabic Argument Mining.

This module handles the creation and processing of datasets for different
argument mining tasks: end-to-end token classification, argument detection,
and argument classification.
"""

import os
from datasets import Dataset, DatasetDict, Sequence, ClassLabel
from .config import (
    END_TO_END_LABELS,
    ARGDET_TOK_LABELS,
    ARGDET_SENT_LABELS,
    ARGCLASS_LABELS,
    SPLITS,
)


def parse_bio_file(file_path: str) -> tuple[list[list[str]], list[list[str]]]:
    """
    Parse a BIO format annotation file.

    Args:
        file_path: Path to the BIO annotation file

    Returns:
        Tuple of (tokens_list, tags_list)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    tokens_list: list[list[str]] = []
    tags_list: list[list[str]] = []
    current_tokens: list[str] = []
    current_tags: list[str] = []

    for line in lines:
        if line.strip() == "" or (
            len(line.split("\t")) >= 2 and line.split("\t")[0] == ""
        ):
            if current_tokens:
                tokens_list.append(current_tokens)
                tags_list.append(current_tags)
                current_tokens = []
                current_tags = []
        else:
            parts = line.split("\t")
            if len(parts) >= 3:
                token = parts[1].strip()
                tag = parts[2].strip()
                current_tokens.append(token)
                current_tags.append(tag)

    if current_tokens:
        tokens_list.append(current_tokens)
        tags_list.append(current_tags)

    return tokens_list, tags_list


def create_end_to_end_dataset(input_dir: str) -> DatasetDict:
    """
    Create dataset for end-to-end argument component identification.

    Args:
        input_dir: Directory containing annotation files
        output_name: Name for the output dataset

    Returns:
        DatasetDict with train, dev, test splits
    """
    data: dict[str, dict[str, list[list[str]]]] = {"train": {}, "dev": {}, "test": {}}

    for split in SPLITS:
        file_path = os.path.join(input_dir, f"{split}-OPUS-annotationsar.txt")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue

        tokens, tags = parse_bio_file(file_path)

        # Normalize tags to ensure consistent format
        normalized_tags: list[list[str]] = []
        for tag_seq in tags:
            normalized_seq: list[str] = []
            for tag in tag_seq:
                if "-MajorClaim" in tag:
                    normalized_seq.append(tag[0] + "-MajorClaim")
                elif "-Claim" in tag:
                    normalized_seq.append(tag[0] + "-Claim")
                elif "-Premise" in tag:
                    normalized_seq.append(tag[0] + "-Premise")
                else:
                    normalized_seq.append(tag)
            normalized_tags.append(normalized_seq)

        data[split]["tokens"] = tokens
        data[split]["tags"] = normalized_tags

    # Create DatasetDict
    dataset_dict = DatasetDict()

    for split in SPLITS:
        if data[split]:  # Only create if data exists
            dataset = Dataset.from_dict(data[split])
            dataset = dataset.cast_column(
                "tags", Sequence(ClassLabel(names=END_TO_END_LABELS))
            )
            dataset_dict[split] = dataset

    return dataset_dict


def create_argument_detection_token_dataset(input_dir: str) -> DatasetDict:
    """
    Create dataset for token-level argument detection.

    Args:
        input_dir: Directory containing annotation files
        output_name: Name for the output dataset

    Returns:
        DatasetDict with train, dev, test splits
    """
    data: dict[str, dict[str, list[list[str]]]] = {"train": {}, "dev": {}, "test": {}}

    for split in SPLITS:
        file_path = os.path.join(input_dir, f"{split}-OPUS-annotationsar.txt")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue

        tokens, tags = parse_bio_file(file_path)

        # Normalize tags to ensure consistent format
        normalized_tags: list[list[str]] = []
        for tag_seq in tags:
            normalized_seq: list[str] = []
            for index, tag in enumerate(tag_seq):
                if tag == "O":
                    normalized_seq.append("O")
                elif tag_seq[index - 1] == "O":
                    normalized_seq.append("B-Arg")
                else:
                    normalized_seq.append("I-Arg")
            normalized_tags.append(normalized_seq)

        data[split]["tokens"] = tokens
        data[split]["tags"] = normalized_tags

    # Create DatasetDict
    dataset_dict = DatasetDict()

    for split in SPLITS:
        if data[split]:
            dataset = Dataset.from_dict(data[split])
            dataset = dataset.cast_column(
                "tags", Sequence(ClassLabel(names=ARGDET_TOK_LABELS))
            )
            dataset_dict[split] = dataset

    return dataset_dict


def create_argument_detection_sentence_dataset(input_dir: str) -> DatasetDict:
    """
    Create dataset for sentence-level argument detection.

    Args:
        input_dir: Directory containing annotation files
        output_name: Name for the output dataset

    Returns:
        DatasetDict with train, dev, test splits
    """
    data: dict[str, dict[str, list[str]]] = {"train": {}, "dev": {}, "test": {}}

    for split in SPLITS:
        file_path = os.path.join(input_dir, f"{split}-OPUS-annotationsar.txt")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue

        tokens, tags = parse_bio_file(file_path)

        # Extract argument sentences and their types
        sentences: list[str] = []
        labels: list[str] = []

        for token_seq, tag_seq in zip(tokens, tags):
            start = 0
            idx = 1
            while idx < len(token_seq):
                if any(ch in token_seq[idx] for ch in (".", "!", "؟")):
                    sentences.append(" ".join(token_seq[start : idx + 1]))
                    if any(t != "O" for t in tag_seq[start : idx + 1]):
                        labels.append("Argumentative")
                    else:
                        labels.append("Non-Argumentative")
                    start = idx + 1
                idx += 1
            if start < len(token_seq):
                sentences.append(" ".join(token_seq[start:]))
                if any(t != "O" for t in tag_seq[start:]):
                    labels.append("Argumentative")
                else:
                    labels.append("Non-Argumentative")

        data[split]["text"] = sentences
        data[split]["label"] = labels

    # Create DatasetDict
    dataset_dict = DatasetDict()

    for split in SPLITS:
        if data[split]:
            dataset = Dataset.from_dict(data[split])
            dataset = dataset.cast_column("label", ClassLabel(names=ARGDET_SENT_LABELS))
            dataset_dict[split] = dataset

    return dataset_dict


def create_argument_classification_dataset(input_dir: str) -> DatasetDict:
    """
    Create dataset for argument component classification.

    Args:
        input_dir: Directory containing annotation files
        output_name: Name for the output dataset

    Returns:
        DatasetDict with train, dev, test splits
    """
    data: dict[str, dict[str, list[str]]] = {"train": {}, "dev": {}, "test": {}}

    for split in SPLITS:
        file_path = os.path.join(input_dir, f"{split}-OPUS-annotationsar.txt")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue

        tokens, tags = parse_bio_file(file_path)

        # Extract argument sentences and their types
        sentences: list[str] = []
        labels: list[str] = []

        for token_seq, tag_seq in zip(tokens, tags):
            start = 0
            idx = 1
            while idx < len(token_seq):
                if any(ch in token_seq[idx] for ch in (".", "!", "؟")):
                    if any(t != "O" for t in tag_seq[start : idx + 1]):
                        sentences.append(" ".join(token_seq[start : idx + 1]))
                        if any("MajorClaim" in t for t in tag_seq[start : idx + 1]):
                            labels.append("MajorClaim")
                        elif any("Claim" in t for t in tag_seq[start : idx + 1]):
                            labels.append("Claim")
                        else:
                            labels.append("Premise")
                    start = idx + 1
                idx += 1
            if start < len(token_seq):
                if any(t != "O" for t in tag_seq[start:]):
                    sentences.append(" ".join(token_seq[start:]))
                    if any("MajorClaim" in t for t in tag_seq[start:]):
                        labels.append("MajorClaim")
                    elif any("Claim" in t for t in tag_seq[start:]):
                        labels.append("Claim")
                    else:
                        labels.append("Premise")

        data[split]["text"] = sentences
        data[split]["label"] = labels

    # Create DatasetDict
    dataset_dict = DatasetDict()

    for split in SPLITS:
        if data[split]:
            dataset = Dataset.from_dict(data[split])
            dataset = dataset.cast_column("label", ClassLabel(names=ARGCLASS_LABELS))
            dataset_dict[split] = dataset

    return dataset_dict


def create_all_datasets(input_dir: str) -> dict[str, DatasetDict]:
    """
    Create all three dataset types.

    Args:
        input_dir: Directory containing annotation files

    Returns:
        Dictionary containing all three datasets
    """
    datasets = {}

    print("Creating end-to-end dataset...")
    datasets["end_to_end"] = create_end_to_end_dataset(input_dir)

    print("Creating argument detection token dataset...")
    datasets["argument_detection_token"] = create_argument_detection_token_dataset(
        input_dir
    )

    print("Creating argument detection sentence dataset...")
    datasets["argument_detection_sentence"] = (
        create_argument_detection_sentence_dataset(input_dir)
    )

    print("Creating argument classification dataset...")
    datasets["argument_classification"] = create_argument_classification_dataset(
        input_dir
    )

    return datasets


if __name__ == "__main__":
    # Example usage
    input_directory = "./data/annotated"

    # Create all datasets
    datasets = create_all_datasets(input_directory)

    # Print dataset info
    for name, dataset in datasets.items():
        print(f"\n{name} dataset:")
        print(dataset)
