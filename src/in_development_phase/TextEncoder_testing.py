import torch

from textEncoders import SmallTextEncoder as TextEncoder

import torch
from transformers import BertTokenizer

"function that tests TextEncoder class"

# 1. Prepare test input
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example batch of text
texts = [
    "draw a circle",
    "move to the left side",
    "rotate and draw another shape"
]

# 2. Tokenize input
encoding = tokenizer(
    texts,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=20
)

input_ids = encoding['input_ids']         # shape: (batch_size, seq_length)
attention_mask = encoding['attention_mask']  # shape: (batch_size, seq_length)

# 3. Initialize model
model = TextEncoder()

# 4. Run forward pass
with torch.no_grad():
    outputs = model(input_ids, attention_mask)

print("Input shape:", input_ids.shape)

print("Output shape:", outputs.shape)  # should be (batch_size, seq_len, d_model)
