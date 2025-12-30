import os
import torch
import torch.nn as nn

# Import your model.py (adjust if path differs)
from dgcnn.pytorch.model import DGCNN

# --------------------
# Args object required by your DGCNN constructor
# --------------------
class Args:
    def __init__(self, k=20, emb_dims=1024, dropout=0.5):
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout


# --------------------
# Wrapper
# --------------------
class DGCNNWrapper(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, num_classes=40):
        super().__init__()
        args = Args(k=k, emb_dims=emb_dims, dropout=dropout)
        self.model = DGCNN(args, output_channels=num_classes)

    def forward(self, x):
        return self.model(x)

    # ---------------------------------------------------------
    # Load PRETRAINED model (.t7, .pth, .pt)
    # ---------------------------------------------------------
    def load_pretrained(self, path, map_location="cpu", strict=True):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=map_location)

        # .t7/.pth common patterns
        if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            raise RuntimeError("Unrecognized checkpoint format.")

        # fix keys (remove DataParallel 'module.' prefix)
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith("module.") else k
            clean_state_dict[new_key] = v

        self.model.load_state_dict(clean_state_dict, strict=strict)
        print(f"[OK] Loaded pretrained DGCNN weights from {path}")

    # ---------------------------------------------------------
    # Freeze ALL model parameters
    # ---------------------------------------------------------
    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False
        print("[OK] All DGCNN weights frozen.")

    # ---------------------------------------------------------
    # Optionally: Freeze everything except final classifier
    # ---------------------------------------------------------
    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if not name.startswith("linear3"):  # final classifier
                param.requires_grad = False
        print("[OK] Backbone frozen; classifier trainable.")


# --------------------
# Convenience loader
# --------------------
def load_pretrained_dgcnn(
    checkpoint="pretrained/model.t7",
    freeze=True,
    k=20, emb_dims=1024, dropout=0.5, num_classes=40,
    device="cpu"
):
    model = DGCNNWrapper(k, emb_dims, dropout, num_classes)
    model.load_pretrained(checkpoint, map_location=device)

    if freeze:
        model.freeze_all()

    model.to(device)
    model.eval()

    return model
