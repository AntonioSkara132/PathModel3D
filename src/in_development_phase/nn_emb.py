import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
from src.textEncoders import TextEncoder

"""LangPathModel
This file contains definition of LangPathModel with Classification
head instead of regression one, classification approach has the
potential to beat regression on generation task
This function doesnt have loss because its supposed to use
Cross Entropy loss
"""


class LangPathModel(nn.Module):
    def __init__(self, 
                 d_traj = 4, 
                 d_model=512, 
                 num_heads_encoder=8,
                 num_heads_decoder=8,
                 num_decoder_layers=5,
                 num_encoder_layers=1,
                 hidden_dim=512, 
                 dropout = 0.1, 
                 max_length=1000):
        super(TrajectoryModel, self).__init__()
        
        # Embedding layer for input and output
        self.inputLayer = nn.Linear(4, d_model)
        
        self.positional_encoding = self.get_positional_encoding(max_length, d_model)
        self.text_encoder = TextEncoder(output_dim=d_model)

        decoderLayer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads_decoder,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first = True	#[B, S, T] vs [S, B, T]
        )
        self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoderLayer, num_layers=num_decoder_layers) #herea is decoder defined
        self.output_position(d_model, 100*100) 	#coordinate embeddings
        self.output_actions(d_model, 2) 		#action embedding
        self.output_stops(d_model, 2) $ 		#stop embedding

    def forward(self, text, path_mask, tgt):
        emb_tgt = self.input_embedding(tgt) 

        emb_src = emb_src + self.positional_encoding[:path_len].permute(1, 0, 2)
        emb_tgt = emb_tgt + self.positional_encoding[:tgt_len].permute(1, 0, 2)
        emb_text = self.text_encoder(text, text_mask)
      
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=path.device) * float('-inf'), diagonal=1).bool() #look-ahead maska

        out = self.decoder(emb_tgt, memory=emb_text, tgt_mask=tgt_mask)
        out = out @ self.embedding..weights.T
        return out
    
    def get_positional_encoding(self, max_length, d_model):
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe

