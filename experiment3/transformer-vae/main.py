import datetime

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from data_processing import preprocess_data, split_data

import numpy as np
import random
# Set a fixed randomness for reproducibility.
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

NUM_EPOCHS = 10000

class TransformerVAE(nn.Module):
    def __init__(self, model_name, latent_dim):
        super(TransformerVAE, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Define the mean and log-variance layers for the latent space
        self.fc_mu = nn.Linear(self.encoder.config.hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.config.hidden_size, latent_dim)

        # Add a linear layer to map hidden states to the vocabulary size
        self.linear_layer = nn.Linear(self.encoder.config.hidden_size, self.tokenizer.vocab_size)

        # 1 dimension: rejected or not
        self.rejection_classifier = nn.Linear( latent_dim, 1 )
        self.sigmoid = nn.Sigmoid()

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

    # def decode_train(self, z, input_seq ):
    #     # Adjust loop to generate tokens up to input_seq_len
    #     nBatches = input_seq.shape[0]
    #     batchLength = input_seq.shape[1]
    #     attention_masks = torch.triu( torch.ones( batchLength, batchLength ) )
    #     out_logits = torch.zeros( ( nBatches, batchLength, self.linear_layer.out_features ) )
    #     for iToken in range( batchLength ):
    #         attention_mask = attention_masks[iToken].unsqueeze(0)
    #         outputs = self.decoder( input_ids = input_seq, encoder_hidden_states = z, 
    #                                 attention_mask = attention_mask ).last_hidden_state
    #         token_weights = self.linear_layer(outputs[:,iToken,:])            
    #         out_logits[:,iToken,:] = torch.softmax( token_weights, dim = -1 )
            
    #     return out_logits
    
    # def decode_train2(self, z, input_seq ):
    #     nBatches = input_seq.shape[0]
    #     batchLength = input_seq.shape[1]
    #     out_logits = torch.zeros( ( nBatches, batchLength, self.linear_layer.out_features ) )
    #     for iToken in range( batchLength ):
    #         outputs = self.decoder( input_ids = input_seq[:iToken + 1], encoder_hidden_states = z ).last_hidden_state
    #         token_weights = self.linear_layer(outputs[:,iToken,:])            
    #         out_logits[:,iToken,:] = torch.softmax( token_weights, dim = -1 )
            
    #     return out_logits
    
    def decode_train_parallel(self, z, input_seq ):
        batchLength = input_seq.shape[1]
        attention_masks = torch.triu( torch.ones( batchLength, batchLength ) ).T
        attention_masks = attention_masks.unsqueeze( 0 )
        outputs = self.decoder( input_ids = input_seq, encoder_hidden_states = z, 
                                attention_mask = attention_masks ).last_hidden_state
        token_logits = self.linear_layer(outputs)
           
        return token_logits

    def decode_infer(self, z, input_seq_len ):
        generated = torch.full((z.size(0), 1), self.tokenizer.cls_token_id, dtype=torch.long, device=z.device)
        logits = None
        # Adjust loop to generate tokens up to input_seq_len
        for _ in range(input_seq_len - 1):  # Adjust for the [CLS] token
            bert_outputs = self.decoder(input_ids=generated, encoder_hidden_states=z).last_hidden_state
            next_token_logits = self.linear_layer(bert_outputs[:, -1:, :])
            if logits is None:
                logits = next_token_logits
            else:
                logits = torch.cat( ( logits, next_token_logits ), dim = 1 )
            # Softmax is unnecessary, because cross-entropy loss performs softmax there.
            next_token = torch.argmax(next_token_logits, dim=-1)
            generated = torch.cat((generated, next_token), dim=-1)
        
        return logits
    
    # def decode_old( self, z, input_seq_len ):
    #     generated = torch.full((z.size(0), 1), self.tokenizer.cls_token_id, dtype=torch.long, device=z.device)

    #     # Adjust loop to generate tokens up to input_seq_len
    #     for _ in range(input_seq_len - 1):  # Adjust for the [CLS] token
    #         if generated.size(1) == input_seq_len:  # Stop if sequence length is reached
    #             break
    #         outputs = self.decoder(input_ids=generated, encoder_hidden_states=z)
    #         hidden_states = outputs.last_hidden_state
    #         next_token_logits = self.linear_layer(hidden_states[:, -1, :])

    #         next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    #         generated = torch.cat((generated, next_token), dim=-1)

    #     logits = self.linear_layer(hidden_states[:, :input_seq_len, :])
    #     return logits        


    def logits_to_tokens( self, logits ):
        return torch.argmax( logits, dim=2 )
        

    def forward(self, input_ids, attention_mask):
        #print("Shape of input_ids in forward:", input_ids.shape)  # Add this line
        mu, logvar = self.encode(input_ids, attention_mask)
        z = self.reparameterize(mu, logvar)
        rejected = self.sigmoid(self.rejection_classifier( z ))
        if self.training:
            # For fast training, if it works:
            # reconstructed = self.decode_train_parallel( z, input_ids )
            # For testing that both modes are legitimate:
            reconstructed = self.decode_infer( z, input_ids.size(1) )
        else:
            reconstructed = self.decode_infer( z, input_ids.size(1) )
        return reconstructed, mu, logvar, rejected

    
    def to(self, device):
        super().to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.fc_mu.to(device)
        self.fc_logvar.to(device)
        self.linear_layer.to(device)  # Don't forget to move this layer as well
        self.rejection_classifier.to(device)
        return self

    def vae_loss(self, reconstructed, input_ids, mu, logvar):
        # Align sequence lengths: pad or truncate `reconstructed` to match `input_ids`
        seq_len = input_ids.size(1)
        current_len = reconstructed.size(1)
    
        if current_len > seq_len:
            # Truncate reconstructed if it's longer
            reconstructed = reconstructed[:, :seq_len, :]

        # Flatten tensors for loss computation
        recon_loss = F.cross_entropy(
            reconstructed.reshape(-1, self.tokenizer.vocab_size),  # [batch_size * seq_len, vocab_size]
            input_ids.reshape(-1),  # [batch_size * seq_len]
            reduction='sum'
        )
        print( f"Recon loss: {recon_loss}")

        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        print( f"KL Div Loss: {kl_div}")
        kl_div *= 0.01
        return recon_loss + kl_div


class MyDataset(Dataset):
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

# Check if MPS is available and use it; otherwise, use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model_name = 'bert-base-uncased'
latent_dim = 512  # Example latent dimension size
vae = TransformerVAE(model_name, latent_dim).to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-1)

# Load and preprocess labeled refusals into train/val/test.
X, y = preprocess_data('all_hand_labeled.json', 'response')
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

#------------------------------------------------
# Use this for quick training with only refusals (easy)
#------------------------------------------------
texts =   [X_train[i][:64] for i in range(len(X_train)) if y_train[i] == 1][:2]
classes = torch.tensor( [ 1. ] * 50 ).float().reshape( ( -1, 1 ) )
#------------------------------------------------
# Use this for thorough training with both classes (hard)
#------------------------------------------------
# texts = X_train[:]
# classes = torch.tensor( y_train ).float().reshape( ( -1, 1 ) )
#------------------------------------------------

# filter to short refusals
print(f"{len(texts)} texts, example: {texts[0]}")

dataset = MyDataset(texts, classes, vae.tokenizer, max_length=16)

# Adjust batch size if necessary
batch_size = min(2, len(dataset))  # Ensure batch size is not larger than dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False) 

mseLoss = nn.MSELoss()

# Training loop
vae.train()
epochStartTime = datetime.datetime.now()
for epoch in range(NUM_EPOCHS):
    try:
        if epoch < 2:
            print("Starting epoch", epoch)
        epoch_loss = 0.0  # Initialize epoch loss
        num_batches = 0  # Initialize numberof batches processed

        vae.train()
        for input_ids, attention_mask, rejected in dataloader:
            num_batches += 1
            input_ids, attention_mask, rejected = input_ids.to(device), attention_mask.to(device), rejected.to(device)
            target_ids = input_ids[:,1:] # Don't expect logits for [CLS] output.
            optimizer.zero_grad()
            reconstructed, mu, logvar, classification = vae(input_ids, attention_mask)
            loss = vae.vae_loss(reconstructed, target_ids, mu, logvar)
            classLoss = mseLoss( rejected, classification)
            print( f"Classification loss: {classLoss}")
            loss += classLoss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        # if epoch % 10 == 0:
        # Avoid division by zero
        epochTime = datetime.datetime.now() - epochStartTime
        epochStartTime = datetime.datetime.now()

        if num_batches > 0:
            average_loss = epoch_loss / num_batches
            print(f"\nEpoch {epoch}, Average Loss: {average_loss:.4f}, Time taken: {epochTime}")
        else:
            print(f"Epoch {epoch}, No batches processed")
        
        
        # Forward pass example
        input_ids, attention_mask, rejected = next(iter(dataloader))
        # Dataloader gives us a batch but we just want one sample for now.
        input_ids, attention_mask, rejected = ( input_ids[:1].to(device), 
                                                attention_mask[:1].to(device), 
                                                rejected[:1].to(device) )
        correct_tokens = [vae.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        print(f"Correct tokens:   { ' '.join( correct_tokens[0] ) }") 
        
        # See the result of encoding and decoding in TRAINING mode.
        vae.train()
        reconstructed, _, _, classification = vae(input_ids, attention_mask)
        tokens = vae.logits_to_tokens( reconstructed )
        predicted_tokens = [vae.tokenizer.convert_ids_to_tokens(ids) for ids in tokens]
        print(f"Predicted (train) tokens: { ' '.join( predicted_tokens[0] ) }" )
        # See the result of encoding and decoding in EVAL (non-training) mode.
        vae.eval()
        reconstructed2, _, _, classification2 = vae(input_ids, attention_mask)
        tokens2 = vae.logits_to_tokens( reconstructed2 )
        predicted_tokens2 = [vae.tokenizer.convert_ids_to_tokens(ids) for ids in tokens2]
        print(f"Predicted (eval) tokens:  { ' '.join( predicted_tokens2[0] ) }" )
        vae.train()
        
        # Predicted classification
        print( f"Class: Expected {rejected.item()}, Training {classification.item():.2f}, Eval {classification2.item():.2f}")
    except KeyboardInterrupt as e:
        print( "WBS: Program interrupted, dropping to debug console.  Type 'c' to resume.")
        import pdb; pdb.set_trace()
