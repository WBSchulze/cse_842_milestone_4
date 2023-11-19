import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, texts, rejected, tokenizer, max_length):
        self.texts = texts
        self.rejected = rejected
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return inputs.input_ids.squeeze(0), inputs.attention_mask.squeeze(0), self.rejected[idx]

# Define how to load and preprocess the data.
def preprocess_data(file_path, text_source):
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    # Filter out unwanted columns
    df = df.drop(columns=['finish_reason'])

    # Filter out unwanted classes
    df = df.loc[~df['tone'].isin(['incoherent', 'dontknow'])].copy()

    # Change any label that isn't "complied" to "rejected"
    # df.loc[~df['tone'].isin(['complied', 'rejected']), 'tone'] = 'rejected'
    df.loc[df['tone'].isin(['complied']), 'tone'] = 0
    df.loc[~df['tone'].isin([0]), 'tone'] = 1

    X = df[text_source].tolist()
    rejected = np.array( df['tone'].tolist() )

    return X, rejected, df

# Define how to split the data into train/val/test.
def split_data(X, y):
    # This yields a 80/10/10 train/validation/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_data(text_start_index, text_length, training_dataset_size, max_tokenized_length, tokenizer):
    # Load and preprocess data, then setup the dataloader.
    X, y, df = preprocess_data('all_hand_labeled.json', 'response')
    print(df, '\n')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    texts =   [X_train[i][text_start_index:text_length] for i in range(len(X_train))][:training_dataset_size]
    print(f"Using {len(texts)} texts.\n")#, example:\n{texts[0]}")
    classes = torch.tensor( [ 1. ] * 50 ).float().reshape( ( -1, 1 ) )
    dataset = CustomDataset(texts, classes, tokenizer, max_length=max_tokenized_length)
    dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True, drop_last=False)

    return dataloader
