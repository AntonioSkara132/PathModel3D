import torch
from LangPathModel.colab_src.nn import TrajectoryModel
from LangPathModel.colab_src.textEncoders import TextEncoder
from torch.optim.lr_scheduler import StepLR

#data = [Batch, sequence, (batch input, batch target)]

def train(model, dataloader, niter, device):
    """Train a decoder‑only model with timing and extra statistics.

    Args:
        model: The neural network to train (already moved to *device*).
        dataloader: Yields (batch_paths, batch_texts).
        niter: Number of epochs.
        device: torch.device (e.g. torch.device("cuda"))
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.2)

    model.train()
    text_encoder = TextEncoder(output_dim=512).to(device)

    for epoch in range(niter):
        # ---------------------------------------------------------------
        #  Start timing this epoch
        # ---------------------------------------------------------------
        epoch_start = time.perf_counter()

        total_loss = 0.0          # Sum of token‑weighted losses
        total_tokens = 0          # Number of target tokens (for avg loss)
        total_grad_norm = 0.0     # For monitoring gradient magnitudes

        for batch_paths, batch_texts in dataloader:
            # -----------------------------------------------------------
            #  Prepare batch data
            # -----------------------------------------------------------
            batch_paths = batch_paths.to(device).float()

            decoder_input = batch_paths[:, :-1]  
            target_output = batch_paths[:, 1:]

            text_ids = batch_texts['input_ids'].to(device)
            text_mask = (batch_texts['attention_mask'] == 0).to(device)

            # -----------------------------------------------------------
            #  Forward / backward pass
            # -----------------------------------------------------------
            optimizer.zero_grad()

            logits = model(
                text=text_ids,
                path=encoder_input,
                path_mask=encoder_input_mask,
                tgt=decoder_input,
                text_mask=text_mask,
            )

            loss = criterion(predictions, target_output)
            loss.backward()

            # Optional: track global gradient norm per batch
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
            total_grad_norm += grad_norm.item()

            optimizer.step()

            # -----------------------------------------------------------
            #  Accumulate statistics
            # -----------------------------------------------------------
            num_tokens = target_output.numel()
            total_tokens += num_tokens
            total_loss += loss.item() * num_tokens  # token‑weighted

        scheduler.step()

        # ---------------------------------------------------------------
        #  Epoch‑level metrics
        # ---------------------------------------------------------------
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        avg_grad_norm = total_grad_norm / len(dataloader)
        epoch_time = time.perf_counter() - epoch_start
        batches_per_sec = len(dataloader) / epoch_time

        print(
            f"Epoch {epoch + 1}/{niter} | "
            f"avg_loss {avg_loss:.4f} | ppl {perplexity:.2f} | "
            f"grad_norm {avg_grad_norm:.2f} | "
            f"time {epoch_time:.2f}s ({batches_per_sec:.2f} batches/s)"
        )

