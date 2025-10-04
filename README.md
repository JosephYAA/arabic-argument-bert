# Arabic Argument Mining with BERT

This project implements cross-lingual argument mining from English persuasive essays to Arabic using BERT models. The pipeline translates English essays annotated in BIO format to Arabic using OPUS-MT, projects the labels, and trains BERT models for four different argument mining tasks.

## Tasks

### 1. End-to-End Argument Component Identification
- **Task**: Token-level classification identifying argument components
- **Labels**: `O`, `B-MajorClaim`, `I-MajorClaim`, `B-Premise`, `I-Premise`, `B-Claim`, `I-Claim`

### 2. Token-Level Argument Detection
- **Task**: Token-level classification identifying argumentative spans
- **Labels**: `O`, `B-Arg`, `I-Arg`

### 3. Sentence-Level Argument Detection
- **Task**: Sentence-level classification identifying argumentative spans
- **Labels**: `Argumenatative`, `Non-Argumentative`

### 4. Argument Component Classification
- **Task**: Multi-class classification of argumentative sentences
- **Labels**: `MajorClaim`, `Claim`, `Premise`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JosephYAA/arabic-argument-mining-bert.git
cd arabic-argument-mining-bert
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Translation Only
```bash
python main.py --mode translate --input_dir ./data/raw --output_dir ./data/translated
```

### Model Training Only
```bash
python main.py --mode train --input_dir ./data/translated
```

### Full Pipeline
```bash
python main.py --mode full --raw_dir ./data/raw --annotated_dir ./data/translated
```

## Project Structure

```
arabic-argument-mining-bert/
├── src/
│   ├── config.py              # Configuration settings
│   ├── translation.py         # OPUS-MT translation functionality, alignment, and projection
│   ├── dataset_processing.py  # Dataset creation for all tasks
│   └── model_training.py      # Model training and evaluation
├── main.py                    # Main script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Data Format

### Input Files
- **Raw English files**: `.dat.abs` format with token-level BIO annotations. You can download these from https://github.com/UKPLab/acl2017-neural_end2end_am/tree/master. Use the paragraph-level set.

### Expected File Structure
```
data/
├── raw/
│   ├── train.dat.abs
│   ├── dev.dat.abs
│   └── test.dat.abs
```

## Model Configuration

The models use the following default configuration:
- **Base Model**: `bert-base-multilingual-uncased`
- **Training Epochs**: 5
- **Token Classification Batch Size**: 8
- **Sequence Classification Batch Size**: 16
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW

These can be modified in `arabic_argument_mining/config.py`.
