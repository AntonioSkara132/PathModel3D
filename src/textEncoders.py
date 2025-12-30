import torch
import torch.nn as nn
from transformers import AutoModel

"""file that stores TextEncoder classes these classes are use BERT to extract semantic features"""

class TextEncoder(nn.Module):
	"""extracts semantic features from text"""
	def __init__(self, 
		model_name='bert-base-uncased', 
		output_dim=128, 
		use_cls_token=False,
		freeze=True,
		dropout=0.1):
		
		super().__init__()
		self.bert = AutoModel.from_pretrained(model_name)
		self.use_cls_token = use_cls_token
		self.dropout = nn.Dropout(dropout)

		hidden_size = self.bert.config.hidden_size
		self.projection = nn.Linear(hidden_size, output_dim, bias=False)

		if freeze:
			for param in self.bert.parameters():
				param.requires_grad = False
				
	def forward(self, input_ids, attention_mask=None, token_type_ids=None):
		outputs = self.bert(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)

		if self.use_cls_token:
			pooled = outputs.last_hidden_state[:, 0]  # CLS token
		else:
			pooled = outputs.last_hidden_state  # Full sequence

		return self.projection(self.dropout(pooled))


