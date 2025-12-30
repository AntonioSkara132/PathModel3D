import torch
from LangPathModel.colab_src.nn_mdn import TrajectoryModel
from LangPathModel.colab_src.textEncoders import TextEncoder
from torch.optim.lr_scheduler import StepLR
from LangPathModel.colab_src import nn
from LangPathModel.colab_src import nn

#data = [Batch, sequence, (batch input, batch target)]

def train(model, dataloader, niter, device):
 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    Scheduler = StepLR(optimizer, step_size = 2, gamma=0.2)

    # The model is already defined outside the train function, no need to redefine it here
    # model = TrajectoryModel()  
    model.train()
    text_encoder = TextEncoder(output_dim=512)
    #text_encoder.to(device) # Move text_encoder to the device

    for epoch in range(niter):
        total_loss = 0
        for batch_paths, batch_texts in dataloader:
            #print(f"paths: {batch_paths[0]}")
            batch_paths = batch_paths.to(device).float()
            #print(type(batch_paths))
            # Shift target for teacher forcing
            decoder_input = batch_paths[:, :-1].to(device)      # all except last token, move to device
            target_output = batch_paths[:, 1:].to(device)        # all except first token, move to device
            encoder_input = batch_paths[:, 0].to(device) # Move encoder_input to the device
            encoder_input_mask = (encoder_input.abs().sum(dim=-1) != 0).int().reshape(-1, 1).to(device) # Move encoder_input_mask to the device
            #emb_text = text_encoder(batch_texts['input_ids'].to(device), batch_texts['attention_mask'].to(device)) # Pass tensors on the device
            text_mask = batch_texts['attention_mask'] == 0
            text = batch_texts['input_ids']

            optimizer.zero_grad()
            #print(f"encoder input mask: {encoder_input_mask.shape}")
            #print(f"encoder_input: {encoder_input.shape}")

            emb_text = emb_text.to(device).float()
            text_mask = text_mask.to(device).bool()

            pi, mu, sigma, logits  = model(text = text, path = encoder_input, path_mask = encoder_input_mask, tgt = decoder_input, text_mask=text_mask)  
	    loss1 = mdn_loss(pi, mu, sigma, target_output[:, 1:, :2)
	    loss2 = classification_loss(logits, target_output[:, 1:, 2:])

	    loss = loss1 + loss2

            #predictions = predictions.reshape(-1, predictions.size(-1))
            #print("predictions:", predictions.shape)        # should be [32, 199, 512]
            #print("target_output:", target_output.shape)    # should be the same

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(loss.item())
        
        Scheduler.step()
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

