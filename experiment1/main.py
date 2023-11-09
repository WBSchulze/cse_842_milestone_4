import torch
import os
from torch.utils.data import TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from data_processing import preprocess_data, split_data

from logistic_regression import logistic_regression
from util import seed_randomness, perturb_and_reconstruct
from variational_autoencoder import VAE

# Set a fixed randomness for reproducibility.
random_seed = 42
seed_randomness(random_seed)

# Load and preprocess labeled refusals into train/val/test.
X, y = preprocess_data('all_hand_labeled.json', 'response')
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Vectorize the data using TF-IDF.
    # Yields a matrix:
        # Each row is a representation of a document (i.e. ChatGPT responses).
        # Each column corresponds to one of the top 5000 words (or less) by term frequency across the corpus.
            # Values in each column are the TF-IDF scores for that word in the document.
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Train a VAE model, or load an existing one.
model_file_path = 'vae_model.pth'
num_features = X_train_tfidf.shape[1] # Vectorizer may decide to use < max_features
model = VAE(input_dim=num_features)
if os.path.isfile(model_file_path):
    model.load_state_dict(torch.load(model_file_path))
else:
    train_data = TensorDataset(
        torch.FloatTensor(X_train_tfidf), # input
        torch.FloatTensor(X_train_tfidf)  # reconstruction target
    )
    model.custom_train(train_data, model_file_path, epochs=5)

# Perturb latent dimensions and print the word reconstructions, then return the perturbed vectors.
z_train, z_test = perturb_and_reconstruct(model, X_train, X_train_tfidf, X_test_tfidf, vectorizer, threshold=0.05, datapoint_index=5)

# Use the perturbed vectors to train a logistic regression model.
logistic_regression(z_train, z_test, y_train, y_test, random_seed)
