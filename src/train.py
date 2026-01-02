import time
from datetime import datetime
import torch
from torch.optim.lr_scheduler import StepLR
from data.augmentation import rotate
import random

""" train of LangPathModel """

def train(
        model,
        dataloader,
        niter,
        device,
        start_lr: float = 1e-4,
        step: int = 10,
        gamma: float = 0.1,
        weight_decay: float = 0.0,
        checkpoint: str | None = None,
        ckpt_every: int = 10,
        ckpt_dir: str = "/checkpoints/"
):
    """Train `model` for `niter` epochs and time each epoch."""

    # optional, load a checkpoint
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint: {checkpoint}")

    model.to(device)
    model.train()                                          
    model.positional_encoding = model.positional_encoding.to(device)

    optimizer  = torch.optim.Adam(model.parameters(),
                                  lr=start_lr,
                                  weight_decay=weight_decay)
    scheduler  = StepLR(optimizer, step_size=step, gamma=gamma)

    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    T = 20

    for epoch in range(1, niter + 1):
        epoch_start = time.time()
        running_loss = 0.0

        for shape, shape_mask in dataloader:

            # text to device + masks
            shape_size = shape.shape[1]
            shape   = shape.to(device).to(torch.float32)
            shape_mask  = shape_mask.to(device)  # True where padding !!!
            #print(shape.item())
            #print(shape_mask[0, :])

            # forward/backward
            optimizer.zero_grad(set_to_none=True)
            random.randint(0, batch_size-1)
            start_idx = torch.randint(0, shape_size - 1, [1])
            start_pt = shape[:, start_idx, :]
            
            preds, occupancy = model(start_pt, shape, shape_mask, T)  
            #print(preds[0, :, :])
  
            loss = model.get_loss(occupancy)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

	# epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss   = running_loss / num_batches
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Epoch {epoch:03d}/{niter} | "
              f"Loss: {avg_loss:8.4f} | "
              f"Occupancy: {occupancy.mean().item():6.4f} | "
              f"Time: {epoch_time:6.2f}s")

        # checkpointing
        if epoch % ckpt_every == 0 or epoch == niter:
            ckpt_path = f"{ckpt_dir}/model_state_epoch_{epoch:03d}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ saved checkpoint to {ckpt_path}")



    
    
