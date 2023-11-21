from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch import nn

BERT_DIM = 768

class GruVae(nn.Module):
    def __init__(self, latent_dim = 512, num_layers = 3):
        super(GruVae, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.embedding = AutoModel.from_pretrained('bert-base-uncased').embeddings
        self.encoder = nn.GRU( BERT_DIM, int( latent_dim / 2 ), num_layers, 
                               bidirectional = True, dropout = 0.0 )

        # Define the mean and log-variance layers for the latent space
        self.fc_mu = nn.Linear( latent_dim, latent_dim )
        self.fc_logvar = nn.Linear( latent_dim, latent_dim )

        # 1 dimension: rejected or not
        self.fc_rejection = nn.Linear( latent_dim, 1 )
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()

        self.decoder = nn.GRU( BERT_DIM, latent_dim, num_layers,
                               bidirectional = False, dropout = 0.1 )
        # Add a linear layer to map hidden states to the vocabulary size
        self.fc_vocab = nn.Linear( latent_dim, self.tokenizer.vocab_size )


    def forward(self, input_ids):
        # BERT embedding requires a batch dimension, which we then squeeze back out.
        embedded = self.embedding( input_ids.unsqueeze(0) ).squeeze(0)
        mu, logvar = self.encode( embedded )
        z = self.reparameterize( mu, logvar )
        rejected = self.sigmoid(self.fc_rejection( z.squeeze() ))
        if self.training:
            reconstructed = self.decode_train( z, embedded )
        else:
            # WBS: Not yet implemented
            reconstructed = self.decode_eval( z )
        return reconstructed, mu, logvar, rejected


    def encode( self, converted ):
        # Encoder returns output, hidden state (unused)
        encoded, _ = self.encoder( converted )
        cls_encoded = encoded[-1:,:]
        mu = self.fc_mu( cls_encoded )
        logvar = self.fc_logvar( cls_encoded )
        return mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def decode_train(self, z, converted ):
        # Decoder returns output, final hidden states (unused)
        hidden_states = torch.cat( (z,z,z), dim = 0 )

        # Dimensions: token, hidden dimension.        
        # Don't want an extra step after last output token.
        input_seq = converted[:-1,:]
        
        decoded, _ = self.decoder( input_seq, hidden_states )
        logits = self.fc_vocab( decoded )
        return logits


    def decode_eval(self, z, max_seq_len = 16 ):
        pre_seq = z
        logits = None
        # # Adjust loop to generate tokens up to input_seq_len
        # for _ in range(max_seq_len - 1):  # Adjust for the [CLS] token
        #     bert_outputs = self.decoder(input_ids=pre_seq, encoder_hidden_states=z)
        #     next_token_logits = self.fc_vocab(bert_outputs[:, -1:, :])
        #     if logits is None:
        #         logits = next_token_logits
        #     else:
        #         logits = torch.cat( ( logits, next_token_logits ), dim = 1 )
        #     # Softmax is unnecessary, because cross-entropy loss performs softmax there.
        #     next_token = torch.argmax(next_token_logits, dim=-1)
        #     pre_seq = torch.cat((pre_seq, next_token), dim=-1)
        
        return logits


    def logits_to_tokens( self, logits ):
        return torch.argmax( logits, dim=-1)
        

    def logits_to_text( self, logits ):
        tokens_int = self.logits_to_tokens( logits )
        tokens_str = [self.tokenizer.convert_ids_to_tokens(ti) for ti in tokens_int]
        tokens_txt = ' '.join( tokens_str[0] )
        return tokens_txt


    def to(self, device):
        super().to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.fc_mu.to(device)
        self.fc_logvar.to(device)
        self.fc_vocab.to(device)
        self.fc_rejection.to(device)
        return self


    def recon_loss( self, reconstructed, input_ids ):
        # Align sequence lengths: pad or truncate `reconstructed` to match `input_ids`
        # Throw away initial [CLS] token when determining right answers.
        target_ids = input_ids[1:]

        # Flatten tensors for loss computation
        loss = F.cross_entropy(
            reconstructed.reshape(-1, self.tokenizer.vocab_size),  # [batch_size * seq_len, vocab_size]
            target_ids.reshape(-1),  # [batch_size * seq_len]
            reduction='sum'
        )
        return loss


    def kl_div_loss(self, mu, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_div
    

    def classification_loss( self, expected, predicted ):
        loss = self.bce_loss( predicted, expected)
        return loss