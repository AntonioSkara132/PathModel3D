
import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
import sys
sys.path.append('/content/LangPathModel/src/')
from textEncoders import TextEncoder
from nn import MDN, CategoricalNetwork

"""LangPathModel
This file contains definition of basic LangPathModel
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
		     max_length=1000,
                     num_components = 10):
		super(LangPathModel, self).__init__()

		self.d_model = d_model
		self.num_encoders = num_encoder_layers
		self.num_decoders = num_decoder_layers
                self.num_components = num_components

		# Embedding layer for input and output
		self.input_embedding = nn.Linear(d_traj, d_model)

		# Positional encoding to add positional information
		self.register_buffer("positional_encoding", self.get_positional_encoding(max_length, d_model))
		self.text_encoder = TextEncoder(output_dim=d_model)

		decoderLayer = torch.nn.TransformerDecoderLayer(
		        d_model=d_model,
		        nhead=num_heads_decoder,
		        dim_feedforward=hidden_dim,
		        dropout=dropout,
		        batch_first = True
		)
		self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoderLayer, num_layers=num_decoder_layers)
		#self.output_layer = nn.Linear(d_model, d_traj)
                self.mdn_head = MDN(d_model, 2, num_components);
                self.action_head = CategoricalNetwork(d_model, 2)
                self.stop_head = CategoricalNetwork(d_model, 2)

	def forward(self, tgt, path_mask, text, text_mask):
		tgt_len = tgt.size(1)

		emb_tgt = self.input_embedding(tgt) 

		emb_tgt = emb_tgt + self.positional_encoding[:tgt_len].permute(1, 0, 2)
		
		emb_text = self.text_encoder(text, ~text_mask)      

		tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)

		out = self.decoder(emb_tgt, memory=emb_text, tgt_mask=tgt_mask, tgt_key_padding_mask = path_mask[:, :-1], memory_key_padding_mask = text_mask)
		pi_net, xy_net = self.mdn_head(out)
		qa = self.action_head(out)
		qs = self.stop_head(out) 
		return pi_net, xy_net, qa, qs

        def get_positional_encoding(self, max_length, d_model):
		position = torch.arange(0, max_length).unsqueeze(1).float()
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
		pe = torch.zeros(max_length, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		return pe
        
        def loss(self, x, targets, path_mask, text, text_mask):
                pi, normal, qa, qs = self.forward(x, path_mask, text, text_mask)
                pos_targets = targets[:, :, :2]
                loglik = normal.log_prob(pos_targets.unsqueeze(-2).expand_as(normal.loc))
                L = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=-1)
                L *= path_mask

                action_targets = targets[:, :, 2]
                stop_targets = targets[:, :, 3]
                
                La = -qa.log_prob(action_targets)
                Ls = -qs.log_prob(stop_targets)
                return L + La + Ls
        
        

    






	