import torch
import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from data_processing import preprocess_data, split_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

# set fixed seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # standard deviation
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# load and preprocess data
X, y = preprocess_data('all_hand_labeled.json', 'response')
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# convert text data to TF-IDF
vectorizer = TfidfVectorizer(max_features=2000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# hyperparameters
input_dim = X_train_tfidf.shape[1]
hidden_dim = 500
latent_dim = 20
batch_size = 16
learning_rate = 1e-3
epochs = 500

model_file_path = 'vae_model.pth'

# if a model file doesn't exist, train a new one
if os.path.isfile(model_file_path):
    model = VAE(input_dim, hidden_dim, latent_dim)
    model.load_state_dict(torch.load(model_file_path))
else:
    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_data = TensorDataset(torch.FloatTensor(X_train_tfidf), torch.FloatTensor(X_train_tfidf))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}')
    
    torch.save(model.state_dict(), model_file_path)

def get_top_tfidf_words(vectorizer, reconstructed_vec, top_n=10):
    reconstructed_dense = reconstructed_vec.numpy()

    # threshold to get binary vector
    reconstructed_binary = (reconstructed_dense > 0.5).astype(int)
    feature_names = np.array(vectorizer.get_feature_names_out())

    # get indices of top tf-idf values
    top_indices = reconstructed_binary.argsort()[-top_n:]

    # map indices to words
    top_words = feature_names[top_indices]
    if reconstructed_binary.ndim == 1:
        top_words = top_words[::-1]

    return top_words

# threshold for which tf-idf values to count as words
threshold = 0.05

# convert reconstructed vectors into words
def vectors_to_words(vectorizer, vec):
    binary_vec = (vec > threshold).astype(int)
    binary_vec_2d = binary_vec.reshape(1, -1)
    # Use vectorizer's inverse transform to convert back to words
    words = vectorizer.inverse_transform(binary_vec_2d)
    return words

# use VAE encoder to get latent representations
model.eval()
with torch.no_grad():
    # encode and get latent variable distributions
    mu_train, logvar_train = model.encode(torch.FloatTensor(X_train_tfidf))
    mu_test, logvar_test = model.encode(torch.FloatTensor(X_test_tfidf))

    # reparameterize to get the latent representations
    z_train = model.reparameterize(mu_train, logvar_train)
    z_test = model.reparameterize(mu_test, logvar_test)

    # decode
    reconstructed_X_train_before = model.decode(z_train)

    # perturb latent dims
    latent_dim_to_zero = 15 # (of 20)
    z_train[:, latent_dim_to_zero] = 0
    z_test[:, latent_dim_to_zero] = 0

    # decode 
    reconstructed_X_train_after = model.decode(z_train)

    i = 5
    original_sentence = X_train[i]
    original_vec = torch.FloatTensor(X_train_tfidf[i]).numpy()
    reconstructed_vec_before = reconstructed_X_train_before[i].cpu().detach().numpy()
    reconstructed_vec_after = reconstructed_X_train_after[i].cpu().detach().numpy()

    # convert vectors to words
    original_words = vectors_to_words(vectorizer, original_vec)
    reconstructed_words_before = vectors_to_words(vectorizer, reconstructed_vec_before)
    reconstructed_words_after = vectors_to_words(vectorizer, reconstructed_vec_after)
    
    print(f"\nOriginal response (i={i}):\n{original_sentence}\n")
    print(f"Reconstruction before zeroing out latent dimension (sample {i}):\n{reconstructed_words_before}\n")
    print(f"Reconstruction after zeroing out latent dimension (sample {i}):\n{reconstructed_words_after}")
    print()

    # find changes in the reconstructed words before and after the change
    words_before_set = set(reconstructed_words_before[0])
    words_after_set = set(reconstructed_words_after[0])

    removed_words = words_before_set - words_after_set
    added_words = words_after_set - words_before_set

    print(f"Words removed by zeroing out latent dimension {latent_dim_to_zero} (sample {i}):\n{sorted(removed_words)}\n")
    print(f"Words added by zeroing out latent dimension {latent_dim_to_zero} (sample {i}):\n{sorted(added_words)}\n")


# prep encoded data for logistic regression
X_train_encoded = z_train.cpu().numpy()
X_test_encoded = z_test.cpu().numpy()

# train logistic regression on VAE encoded data
lr_model = LogisticRegression(max_iter=10000, random_state=random_seed)
lr_model.fit(X_train_encoded, y_train)

# get test accuracy
y_pred = lr_model.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_pred)
print(f'VAE + Logistic Regression Test Accuracy: {accuracy*100:.2f}%')
