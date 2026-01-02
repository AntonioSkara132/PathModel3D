import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
import sys
sys.path.append('/content/LangPathModel/src/')
from src.pointnet import PointNet as ShapeEncoder

"""LangPathModel
This file contains definition of basic LangPathModel
"""


class PathModel3D(nn.Module):
        def __init__(self, 
		 d_traj = 3, 
		 d_model=512, 
		 num_heads_encoder=8,
		 num_heads_decoder=8,
		 num_decoder_layers=5,
                 hidden_dim=512, 
                 dropout = 0, 
                 max_length=1000):
                super(PathModel3D, self).__init__()

                self.d_model = d_model
                self.d_traj = d_traj
                self.num_decoders = num_decoder_layers

                # Positional encoding to add positional information
                pe = self.get_positional_encoding(max_length, d_model)
                self.register_buffer("positional_encoding", pe)
                
                self.shape_encoder =  self.shape_encoder = ShapeEncoder(
                        emb_dims=d_model,
                        input_shape="bnc",
                        global_feat=True
                )

                decoderLayer = torch.nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=num_heads_decoder,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    batch_first = True
                )
                self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoderLayer, num_layers=5)
                self.output_layer = nn.Linear(d_model, d_traj)

        def forward(self, start_pt, shape, shape_mask, tgt_len):
                B = shape.shape[0]
                #print(f"B: {B}")
                emb_tgt = torch.zeros([B, tgt_len, self.d_model], device = shape.device)
                #print(f"emb_tgt dims: {emb_tgt.shape}")
                #print(f"shape dims: {shape.shape}")     

                emb_tgt = emb_tgt + self.positional_encoding[:tgt_len].permute(1, 0, 2)
                
                emb_shape = self.shape_encoder(shape).permute(0, 2, 1)
                #print(f"emb_shape dims: {emb_shape.shape}")

                tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=shape.device) * float('-inf'), diagonal=1).bool()

                out = self.decoder(emb_tgt, memory=emb_shape, tgt_mask=tgt_mask, memory_key_padding_mask = shape_mask)
                out = self.output_layer(out)
                #print(f"out dims: {out.shape}")
                occupancy = self.update_occupancy(start_pt, out, shape)
                return out, occupancy

        def update_occupancy(self, start_pt, vels, shape, sigma=0.1):
                """
                path: [B, T, 3]
                shape: [B, N, 3]
                """
                assert len(start_pt.shape) == 3
                assert len(vels.shape) == 3
                assert len(shape.shape) == 3

                #print(f"start pt: {start_pt.shape}")

                # squared distances
                path = start_pt + torch.cumsum(vels, dim = 1)
                diffs = (shape[:, None, :, :] - path[:, :, None, :]) ** 2  # [B, T, N, 3]
                d2 = diffs.sum(-1)  # [B, T, N]
                
                # Gaussian contributions
                contribs = torch.exp(-d2 / sigma**2)  # [B, T, N]
                
                # cumulative occupancy (stepwise)
                occupancy = torch.clamp(torch.sum(contribs, dim=1), 0, 1) # or tanh or clamp
                return occupancy
                
        def get_positional_encoding(self, max_length, d_model):
                position = torch.arange(0, max_length).unsqueeze(1).float()
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
                pe = torch.zeros(max_length, 1, d_model)
                pe[:, 0, 0::2] = torch.sin(position * div_term)
                pe[:, 0, 1::2] = torch.cos(position * div_term)
                return pe

        def get_loss(self, occupancy):
                return -torch.log(occupancy.mean())






