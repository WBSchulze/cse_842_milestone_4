import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Define how to load and preprocess the data.
def preprocess_data(file_path, text_source):
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    # Filter out unwanted classes
    df = df.loc[~df['tone'].isin(['incoherent', 'dontknow'])].copy()

    # Change any label that isn't "complied" to "rejected"
    # df.loc[~df['tone'].isin(['complied', 'rejected']), 'tone'] = 'rejected'
    df.loc[df['tone'].isin(['complied']), 'tone'] = 0
    df.loc[~df['tone'].isin([0]), 'tone'] = 1

    X = df[text_source].tolist()
    rejected = np.array( df['tone'].tolist() )

    return X, rejected

# Define how to split the data into train/val/test.
def split_data(X, y):
    # This yields a 80/10/10 train/validation/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
