import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, texts, rejected, confidence, tokenizer, device = 'cpu'):
        self.texts = texts
        self.rejected = rejected
        self.confidence = confidence
        self.tokenizer = tokenizer
        self.device = device
        self._tokenized = dict()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx not in self._tokenized:
            text = self.texts[idx]
            tokenized = self.tokenizer( text, return_tensors="pt")
            self._tokenized[idx] = tokenized.input_ids.squeeze(0).to( self.device )
        return self._tokenized[idx], self.rejected[idx], self.confidence[idx]


# Define how to load and preprocess the data.
def preprocess_data(file_path, text_source, confidence, max_length = 256):
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    # Filter out excessively long prompts
    df = df.loc[df['prompt'].str.len() <= max_length ]

    # Filter out unwanted classes
    df = df.loc[~df['tone'].isin(['incoherent', 'dontknow'])].copy()

    # Change any label that isn't "complied" to "rejected"
    # df.loc[~df['tone'].isin(['complied', 'rejected']), 'tone'] = 'rejected'
    df.loc[df['tone'].isin(['complied']), 'tone'] = 0
    df.loc[~df['tone'].isin([0]), 'tone'] = 1

    X = df[text_source].tolist()
    rejected = torch.Tensor( df['tone'].tolist() ).reshape( (-1,1) )
    confidences = torch.Tensor( [ confidence ] * len( X ) ).reshape( (-1,1) )
    return X, rejected, confidences

# Define how to split the data into train/val/test.
def split_data(X, y, c):
    # This yields a 80/10/10 train/validation/test split
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, c, test_size=0.2, random_state=0)
    X_val, X_test, y_val, y_test, c_val, c_test = train_test_split(X_test, y_test, c_test, test_size=0.5, random_state=0)

    return X_train, X_val, X_test, y_train, y_val, y_test, c_train, c_val, c_test
