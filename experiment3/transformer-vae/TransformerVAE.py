from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch import nn

class TransformerVAE(nn.Module):
    def __init__(self, model_name, latent_dim, dropout_rate=0.1):
        super(TransformerVAE, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Define the mean and log-variance layers for the latent space with dropout
        self.fc_mu = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, latent_dim),
            self.dropout
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, latent_dim),
            self.dropout
        )

        # Add a linear layer to map hidden states to the vocabulary size
        self.linear_layer = nn.Linear(self.encoder.config.hidden_size, self.tokenizer.vocab_size)

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

        # Classifier for rejection with dropout
        self.rejection_classifier = nn.Sequential(
            nn.Linear(latent_dim, 1),
            self.dropout,
            self.sigmoid
        )

        # Decoder can be another transformer or a different architecture
        self.decoder = AutoModel.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids, attention_mask=attention_mask)[0]
        cls_encoded = encoded[:,0,:]
        mu = self.fc_mu(cls_encoded)
        logvar = self.fc_logvar(cls_encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, input_seq_len, temperature=1.0):
        generated = torch.full((z.size(0), 1), self.tokenizer.cls_token_id, dtype=torch.long, device=z.device)
        for _ in range(input_seq_len - 1):
            outputs = self.decoder(input_ids=generated, encoder_hidden_states=z)
            hidden_states = outputs.last_hidden_state
            next_token_logits = self.linear_layer(hidden_states[:, -1, :])
            next_token_logits = next_token_logits / temperature  # Apply temperature
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)  # Sample from the probability distribution
            generated = torch.cat((generated, next_token), dim=-1)
        logits = self.linear_layer(hidden_states[:, :input_seq_len, :])
        return logits

    def logits_to_tokens( self, logits ):
        return torch.argmax( logits, dim=2)

    def forward(self, input_ids, attention_mask):
        mu, logvar = self.encode(input_ids, attention_mask)
        z = self.reparameterize(mu, logvar)
        rejected = self.sigmoid(self.rejection_classifier(z))
        input_seq_len = input_ids.size(1)
        reconstructed = self.decode(z, input_seq_len, temperature=0.7)

        return reconstructed, mu, logvar, rejected
    
    def to(self, device):
        super().to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.fc_mu.to(device)
        self.fc_logvar.to(device)
        self.linear_layer.to(device)
        self.rejection_classifier.to(device)
        return self

    def vae_loss(self, reconstructed, input_ids, mu, logvar):
        # Align sequence lengths: pad or truncate `reconstructed` to match `input_ids`
        seq_len = input_ids.size(1)
        current_len = reconstructed.size(1)

        if current_len < seq_len:
            # Pad reconstructed if it's shorter
            padding = torch.zeros((reconstructed.size(0), seq_len - current_len, reconstructed.size(2)), device=reconstructed.device)
            reconstructed = torch.cat([reconstructed, padding], dim=1)
        elif current_len > seq_len:
            # Truncate reconstructed if it's longer
            reconstructed = reconstructed[:, :seq_len, :]

        # Flatten tensors for loss computation
        recon_loss = F.cross_entropy(
            reconstructed.reshape(-1, self.tokenizer.vocab_size),  # [batch_size * seq_len, vocab_size]
            input_ids.reshape(-1),  # [batch_size * seq_len]
            reduction='sum'
        )

        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kl_div
