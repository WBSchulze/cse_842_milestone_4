from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch import nn

BERT_DIM = 768

class GruVae(nn.Module):
    def __init__(self, latent_dim = 512, num_layers = 3, freeze_embeddings = None):
        super(GruVae, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.embedding = AutoModel.from_pretrained('bert-base-uncased').embeddings
        self.frozen_embeddings = freeze_embeddings
        if self.frozen_embeddings == True:
            self.freeze_embeddings( True ) # Freeze Embeddings
        elif self.frozen_embeddings == False:
            self.freeze_embeddings( False ) # Unfreeze Embeddings

        self.converter = nn.Linear( BERT_DIM, latent_dim )
        self.encoder = nn.GRU( latent_dim, int( latent_dim / 2 ), num_layers, 
                               bidirectional = True, dropout = 0.0 )

        # Define the mean and log-variance layers for the latent space
        self.fc_mu = nn.Linear( latent_dim, latent_dim )
        self.fc_logvar = nn.Linear( latent_dim, latent_dim )

        # 1 dimension: rejected or not
        self.fc_rejection = nn.Linear( latent_dim, 1 )
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()

        self.decoder = nn.GRU( latent_dim, latent_dim, num_layers,
                               bidirectional = False, dropout = 0.1 )
        # Add a linear layer to map hidden states to the vocabulary size
        self.fc_vocab = nn.Linear( latent_dim, self.tokenizer.vocab_size )


    def freeze_embeddings( self, freeze ):
        for params_name, params in self.embedding.named_parameters():
            operation = "Disabling" if freeze else "Enabling"
            print( f"{operation} gradient for {params_name}.")
            params.require_grad = ( not freeze )


    def forward(self, input_ids):
        # BERT embedding requires a batch dimension, which we then squeeze back out.
        # tokens_str = self.tokenizer.convert_ids_to_tokens(input_ids)
        # debug_txt = ' '.join( [ f"{t[1]}({str(t[0].item())})" for t in zip( input_ids, tokens_str ) ] ) 
        # print( f"Input text: {debug_txt}" )
        embedded = self.embedding( input_ids.unsqueeze(0) ).squeeze(0)
        converted = self.converter( embedded )
        mu, logvar = self.encode( converted )
        z = self.reparameterize( mu, logvar )
        rejected = self.sigmoid(self.fc_rejection( z.squeeze() ))
        
        if self.training:
            # reconstructed = self.decode_eval( z, len( input_ids ) )
            reconstructed = self.decode_train( z, converted )
            # reconstructed = self.decode_debug( z, converted )
        else:
            # WBS: Not yet implemented
            # reconstructed = self.decode_eval( z, len( input_ids ) )
            reconstructed = self.decode_train( z, converted )
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


    def decode_debug(self, z, converted ):
        self.eval()
        train_output = self.decode_train( z, converted )
        output_ids = torch.argmax( train_output, dim = -1 )
        tokens_str = self.tokenizer.convert_ids_to_tokens(output_ids)
        debug_txt = ' '.join( [ f"{t[1]}({str(t[0].item())})" for t in zip( output_ids, tokens_str ) ] ) 
        print( f"Parallel Output text: {debug_txt}" )

        hidden_states = torch.cat( (z,z,z), dim = 0 )
        next_input = torch.full( (1,1), self.tokenizer.cls_token_id, device = z.device )
        logits = []
        outputs = []
        while True:
            next_embedded = self.embedding( next_input )
            next_converted = self.converter( next_embedded )
            next_outputs, hidden_states = self.decoder( next_converted.squeeze(0), hidden_states )
            next_logits = self.fc_vocab( next_outputs )
            logits.append( next_logits )
            next_input = torch.argmax( next_logits, dim = -1 ).unsqueeze(0)

            outputs.append( next_input )
            if next_input == self.tokenizer.sep_token_id:
                break
            if len( outputs ) == converted.shape[0] - 1:
                break
        logits = torch.cat( logits, dim = 0 )
        seq_ids = torch.argmax( logits, dim = -1 )
        tokens_str = self.tokenizer.convert_ids_to_tokens(seq_ids)
        seq_txt = ' '.join( [ f"{t[1]}({str(t[0].item())})" for t in zip( seq_ids, tokens_str ) ] ) 
        print( f"Sequential Output text: {seq_txt}" )

        # Sequential with teacher forcing
        hidden_states = torch.cat( (z,z,z), dim = 0 )
        next_converted = converted[:1,:]
        logits = []
        outputs = []
        while True:
            next_outputs, hidden_states = self.decoder( next_converted, hidden_states )
            next_logits = self.fc_vocab( next_outputs )
            logits.append( next_logits )
            next_input = torch.argmax( next_logits, dim = -1 ).unsqueeze(0)
            next_outputs = converted[len(outputs):len(outputs) + 1, :]
            if len(logits) == converted.shape[0] - 1:
                break
        logits = torch.cat( logits, dim = 0 )
        seq_ids = torch.argmax( logits, dim = -1 )
        tokens_str = self.tokenizer.convert_ids_to_tokens(seq_ids)
        seq_txt = ' '.join( [ f"{t[1]}({str(t[0].item())})" for t in zip( seq_ids, tokens_str ) ] ) 
        print( f"Sequential (Teacher Forcing) Output text: {seq_txt}" )
        import pdb; pdb.set_trace()
        return logits


    def decode_eval(self, z, max_seq_len = 16 ):
        hidden_states = torch.cat( (z,z,z), dim = 0 )
        next_input = torch.full( (1,1), self.tokenizer.cls_token_id, device = z.device )
        logits = []
        outputs = []
        while True:
            next_embedded = self.embedding( next_input )
            next_outputs, hidden_states = self.decoder( next_embedded.squeeze(0), hidden_states )
            next_logits = self.fc_vocab( next_outputs )
            logits.append( next_logits )
            next_input = torch.argmax( next_logits, dim = -1 ).unsqueeze(0)
            outputs.append( next_input )
            if next_input == self.tokenizer.sep_token_id:
                break
            if len( outputs ) == max_seq_len - 1:
                break
        logits = torch.cat( logits, dim = 0 )
        
        return logits

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
        # 
        # return logits


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
        # Truncate target IDs to reconstructed values
        target_ids = target_ids[:len(reconstructed)]
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