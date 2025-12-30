import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
from src.textEncoders import TextEncoder

"""LangPathModel
This file contains definition of LangPathModel with Mixed Density
Network at the end, this model should be able to fit better on data
with fewer text annotation and greater data variance in the same 
annotation
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
                 dropout = 0, 
                 max_length=1000):
        super(TrajectoryModel, self).__init__()

        self.d_model = d_model
        self.num_encoders = num_encoder_layers
        self.num_decoders = num_decoder_layers
        
        # Embedding layer for input and output
        self.input_embedding = nn.Linear(d_traj, d_model)
        #self.output_embedding = nn.Linear(d_traj, d_model)
        #self.text_encoder = TextEncoder(output_dim = d_model)
        
        # Positional encoding to add positional information
        self.positional_encoding = self.get_positional_encoding(max_length, d_model)
        self.text_encoder = TextEncoder(output_dim=d_model)

        # encoderLayer = torch.nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=num_heads_encoder,
        #     dim_feedforward=hidden_dim,
        #     dropout=dropout,
        #     batch_first = True
        # )
        # self.encoder = torch.nn.TransformerEncoder(encoder_layer=encoderLayer, num_layers=1)

        decoderLayer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads_decoder,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first = True
        )
        self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoderLayer, num_layers=5)
        # Define M encoder layersA
        #self.encoder_layers = nn.ModuleList(
        #    [EncoderLayer(d_model=d_model, num_heads=num_heads_encoder, dff=hidden_dim, dropout=dropout) for _ in range(num_encoder_layers)]
        #)

        # Define N decoder layers
        #self.decoder_layers = nn.ModuleList(
        #    [DecoderLayer(d_model=d_model, num_heads=num_heads_decoder, dff=hidden_dim, dropout=dropout) for _ in range(num_decoder_layers)]
        #)
        self.output_layer = MDNHead(d_model, 20)
    def forward(self, tgt, path_mask, text, text_mask):
	tgt_len = tgt.size(1)

       	emb_tgt = self.input_embedding(tgt) 

	emb_tgt = emb_tgt + self.positional_encoding[:tgt_len].permute(1, 0, 2)
		
	emb_text = self.text_encoder(text, ~text_mask)      

	tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device) * float('-inf'), diagonal=1).bool()

	out = self.decoder(emb_tgt, memory=emb_text, tgt_mask=tgt_mask, tgt_key_padding_mask = path_mask[:, :-1], memory_key_padding_mask = text_mask)
	out = self.output_layer(out)
	return out

    def forward(self, path, text, path_mask, tgt, text_mask):
        batch_size, path_len = path.size(0), path.size(1)
        tgt_len = tgt.size(1)
        #print(path_len)
        #print(path.shape) 
        #print(tgt.shape)
        #print(tgt.size(1))
        # Embedding
        #print(f"encoding: {self.positional_encoding[:path_len].permute(1, 0, 2).shape}")
        #print(f"encoding: {self.positional_encoding[:tgt_len].unsqueeze(0).shape}")
        emb_src = self.input_embedding(path)
        emb_tgt = self.input_embedding(tgt) 

        emb_src = emb_src + self.positional_encoding[:path_len].permute(1, 0, 2)
        emb_tgt = emb_tgt + self.positional_encoding[:tgt_len].permute(1, 0, 2)
        emb_text = self.text_encoder(text, text_mask).unsqueeze(1)
      
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=path.device) * float('-inf'), diagonal=1).bool()

        out = self.decoder(emb_tgt, memory=emb_text, tgt_mask=tgt_mask)
        
        pi, mu, sigma, logits = self.output_layer(out)
        return out
    
    def get_positional_encoding(self, max_length, d_model):
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe

def classification_loss(logits, target_as):
    logits = logits.view(-1, 2)
    target_as = target_as.view(-1, 2)
    loss_a = nn.functional.binary_cross_entropy_with_logits(logits[:, 0], target_as[:, 0].float())
    loss_s = nn.functional.binary_cross_entropy_with_logits(logits[:, 1], target_as[:, 1].float())
    return loss_a + loss_s
    
class MDNHead(nn.Module):
    def __init__(self, hidden_dim, num_components):
        super(MDNHead, self).__init__()
        self.num_components = num_components
        self.pi = nn.Linear(hidden_dim, num_components)
        self.mu = nn.Linear(hidden_dim, num_components * 2)  # 2 for x and y
        self.sigma = nn.Linear(hidden_dim, num_components * 2)

    def forward(self, x):
        pi = self.pi(x)
        pi = torch.softmax(pi, dim=-1)  # Mixing coefficients

        mu = self.mu(x)
        mu = mu.view(-1, self.num_components, 2)  # Means for x and y

        sigma = self.sigma(x)
        sigma = torch.exp(sigma)  # Ensure positivity
        sigma = sigma.view(-1, self.num_components, 2)  # Std devs for x and y

        return pi, mu, sigma
        
def mdn_loss(pi, mu, sigma, target):
    # target: [batch_size, 2]
    target = target.unsqueeze(1).expand_as(mu)  # [batch_size, num_components, 2]
    prob = (1.0 / (2 * np.pi * sigma.prod(dim=2))) * \
           torch.exp(-0.5 * ((target - mu) ** 2 / sigma ** 2).sum(dim=2))
    weighted_prob = pi * prob
    loss = -torch.log(weighted_prob)
    
def classification_loss(logits, target_as):
    logits = logits.view(-1, 2)
    target_as = target_as.view(-1, 2)
    loss_a = nn.functional.binary_cross_entropy_with_logits(logits[:, 0], target_as[:, 0].float())
    loss_s = nn.functional.binary_cross_entropy_with_logits(logits[:, 1], target_as[:, 1].float())
    return loss_a + loss_s

