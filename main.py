#!/usr/bin/env python3
"""
Main script for Arabic Argument Mining with BERT.

This script demonstrates the complete pipeline:
1. Translation of English essays to Arabic using OPUS-MT
2. Dataset creation for different argument mining tasks
3. Model training and evaluation

Usage:
    python main.py --mode translate --input_dir ./data/raw --output_dir ./data/translated
    python main.py --mode create_datasets --input_dir ./data/annotated
    python main.py --mode train --input_dir ./data/annotated
    python main.py --mode full --raw_dir ./data/raw --annotated_dir ./data/annotated
"""

import argparse
import os
import sys
from pathlib import Path

from src.dataset_processing import create_all_datasets
from src.model_training import train_all_models
from src.translation import translate_dataset

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import SPLITS


def main():
    """Main function to run the Arabic argument mining pipeline."""
    parser = argparse.ArgumentParser(
        description="Arabic Argument Mining with BERT Pipeline"
    )
    _ = parser.add_argument(
        "--mode",
        choices=["translate", "train", "full"],
        required=True,
        help="Pipeline mode to run",
    )
    _ = parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory for dataset creation or training",
    )
    _ = parser.add_argument(
        "--output_dir", type=str, help="Output directory for translations"
    )
    _ = parser.add_argument(
        "--raw_dir", type=str, help="Directory with raw English files (for full mode)"
    )
    _ = parser.add_argument(
        "--annotated_dir",
        type=str,
        help="Directory with annotated Arabic files (for full mode)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Arabic Argument Mining with BERT")
    print("=" * 60)

    if args.mode == "translate":
        if not args.input_dir or not args.output_dir:
            print("Error: --input_dir and --output_dir required for translate mode")
            sys.exit(1)

        print("\n1. TRANSLATION PHASE")
        print("-" * 30)
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Translate all dataset splits
        translate_dataset(
            args.input_dir,
            args.output_dir,
            SPLITS,
        )
        print("\n✓ Translation completed successfully!")

    elif args.mode == "train":
        if not args.input_dir:
            print("Error: --input_dir required for train mode")
            sys.exit(1)

        print("\n3. MODEL TRAINING PHASE")
        print("-" * 30)
        print(f"Input directory: {args.input_dir}")

        # Create datasets first
        datasets = create_all_datasets(args.input_dir)

        # Train all models
        models = train_all_models(datasets)

        print("\n✓ Model training completed successfully!")
        print("\nTrained models:")
        for task_name in models:
            print(f"  - {task_name.replace('_', ' ').title()}")

    elif args.mode == "full":
        if not args.raw_dir or not args.annotated_dir:
            print("Error: --raw_dir and --annotated_dir required for full mode")
            sys.exit(1)

        print("\nFULL PIPELINE EXECUTION")
        print("-" * 30)

        # Step 1: Translation
        print("\n1. TRANSLATION PHASE")
        print("-" * 30)
        os.makedirs(args.annotated_dir, exist_ok=True)
        translate_dataset(args.raw_dir, args.annotated_dir, SPLITS)
        print("✓ Translation completed!")

        # Step 2: Dataset Creation
        print("\n2. DATASET CREATION PHASE")
        print("-" * 30)
        datasets = create_all_datasets(args.annotated_dir)

        print("\nDataset summaries:")
        for name, dataset in datasets.items():
            print(f"\n{name.replace('_', ' ').title()}:")
            for split in dataset:
                print(f"  {split}: {len(dataset[split])} examples")
        print("✓ Dataset creation completed!")

        # Step 3: Model Training
        print("\n3. MODEL TRAINING PHASE")
        print("-" * 30)
        models = train_all_models(datasets)

        print("\n✓ Model training completed!")
        print("\nTrained models:")
        for task_name in models:
            print(f"  - {task_name.replace('_', ' ').title()}")

    print("\n" + "=" * 60)
    print("Pipeline execution completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
