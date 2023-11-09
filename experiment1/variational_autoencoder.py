import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim

        # Hyperparameters
        self.hidden_dim = 500
        self.latent_dim = 20
        self.batch_size = 16
        self.learning_rate = 1e-3

        # Encoding layers (learn the mean and log variance of the latent space)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)       # Input to hidden layer
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)     # Hidden to latent mean
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)     # Hidden to latent log variance

        # Decoding layers (reconstruct the input from the latent space)
        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)      # Latent to hidden layer
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)       # Hidden to reconstructed input layer

    def encode(self, x):
        h1 = F.relu(self.fc1(x))    # fc1  layer's output after ReLU
        means = self.fc21(h1)       # fc21 layer's output (means of latent space)
        logvars = self.fc22(h1)     # fc22 layer's output (log variances of latent space)
        return means, logvars

    def decode(self, z):
        h3 = F.relu(self.fc3(z))                        # fc3 layer's output after ReLU
        reconstruction = torch.sigmoid(self.fc4(h3))    # fc4 layer's output after sigmoid (bounds to [0, 1])
        return reconstruction

    def sample_from_latent_distribution(self, means, logvars):
        stds = torch.exp(0.5 * logvars)     # logvar is log(sigma^2), so std is sigma (0.5 is for square root)
        epsilons = torch.randn_like(stds)   # Vector of random numbers from a standard normal distribution N(0, 1)
        noises = stds * epsilons
        sample = means + noises             # Shift the means according to the noises
        return sample
    
    def get_latent_encoding(self, x):
        means, _ = self.encode(torch.FloatTensor(x))
        return means

    def forward(self, x):
        # Get means and logvars
        means, logvars = self.encode(
            x.view(-1, self.input_dim) # Flatten original data to match decoder's output shape
        ) 
        
        sample = self.sample_from_latent_distribution(means, logvars)   # Sample from resulting latent distribution
        return self.decode(sample), means, logvars                      # Return reconstruction, means, and logvars

    def calculate_loss(self, reconstruction, original, means, logvars):
        # Use binary cross entropy to measure the difference between the original data and the reconstruction.
        binary_cross_entropy = F.binary_cross_entropy(
            reconstruction, 
            original.view(-1, self.input_dim), # Flatten original data to match decoder's output shape
            reduction='sum' # Summing loss over all elements of the batch is a common for VAE loss functions
        )

        # Use KL divergence to penalize divergences of the distribution ~(means, logvars) from the standard normal
        # distribution.
        KL_divergence = -0.5 * torch.sum(1 + logvars - means.pow(2) - logvars.exp())

        loss =  (
            binary_cross_entropy    # Reconstruction loss.
          + KL_divergence           # Loss for deviating the latent space from a standard normal distribution.
        )

        return loss

    def custom_train(self, train_data, model_save_path, epochs):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate) # Updates weights + manages learning rate
        
        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True # Randomize ordering of feeding in datapoints for each epoch
        )

        for epoch in range(epochs):
            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                optimizer.zero_grad()                                           # Reset gradients to 0
                recon_batch, means, logvars = self(data)                        # Forward pass

                # Calculate loss, combining reconstruction loss and standard normal distribution divergence loss
                loss = self.calculate_loss(recon_batch, data, means, logvars)

                loss.backward()                                                 # Backward pass
                train_loss += loss.item()
                optimizer.step()                                                # Update weights
            print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}')

        torch.save(self.state_dict(), model_save_path)
