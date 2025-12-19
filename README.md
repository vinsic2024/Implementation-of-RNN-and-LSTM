# Implementation-of-RNN-and-LSTM

Implementation of RNN and LSTM models for news article topic and political ideology classification using PyTorch.

---

## Overview

This repository implements recurrent neural network (RNN)–based models, including LSTM, for classifying news articles.  
The system performs **two classification tasks**:

### 1. Topic Classification (5 classes)
- Politics (1)
- Entertainment (2)
- Sports (3)
- Technology (4)
- Economics (5)

### 2. Ideology Classification
- Left (-1)
- Neutral (0)
- Right (1)
- Not Political (99)

---

## Files

- **`train.py`**  
  Training script for RNN/LSTM models using text files and ground-truth labels.

- **`inference.py`**  
  Inference script that loads a trained model and predicts topic and ideology labels for a single text file.

- **`best_lstm.pt`**  
  Pretrained LSTM model checkpoint used for inference.

- **`Training Codes APPENDIX Assignment 1 Implementation of RNN and LSTM`**  
  Training code appendix included as part of the assignment report.

- **`Inference Codes APPENDIX Assignment 1 Implementation of RNN and LSTM`**  
  Inference code appendix included as part of the assignment report.

---

## Training

The model is trained using:
- A directory containing news article text files
- A CSV file containing ground-truth labels

The training process automatically splits the dataset into training and validation sets and saves the best-performing model.

### Example training command

```bash
python train.py \
  --text_dir assignment3-1_train_unzipped/assignment3-1_train \
  --csv_path ground_truth_train.csv \
  --out_dir outputs
