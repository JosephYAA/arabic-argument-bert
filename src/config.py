"""
Configuration settings for Arabic Argument Mining project.
"""

# Model settings
# BERT_MODEL_NAME = "bert-base-multilingual-uncased"
BERT_MODEL_NAME = "aubmindlab/bert-base-arabertv2"
NUM_TRAIN_EPOCHS = 5
TOKEN_BATCH_SIZE = 8
SEQUENCE_BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# Label schemes
END_TO_END_LABELS = [
    "O",
    "B-MajorClaim",
    "I-MajorClaim",
    "B-Premise",
    "I-Premise",
    "B-Claim",
    "I-Claim",
]
ARGDET_TOK_LABELS = ["O", "B-Arg", "I-Arg"]
ARGDET_SENT_LABELS = ["Argumentative", "Non-Argumentative"]
ARGCLASS_LABELS = ["MajorClaim", "Claim", "Premise"]

# Dataset splits
SPLITS = ["train", "dev", "test"]

# File paths
OUTPUT_DIR = "./output"
