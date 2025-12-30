import os
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class ShapeDataset(Dataset):
    def __init__(self, directory):
        filenames = os.listdir(directory)
        self.names = [name.removesuffix() for name in filenames]

    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        shape = np.load(os.join(directory, self.filenames[idx])
        return torch.from_numpy(shape)

def collate_fn(batch):
    shapes = batch

    shapes_lengths = torch.tensor([p.size(0) for p in shapes]) 

    padded_shapes = pad_sequence(shapes, batch_first=True, padding_value=0)
    #print(path_lengths)
    max_len = padded_shapes.size(1)
    range_row = torch.arange(max_len, device=path_lengths.device)[None, :]  # shape (1, T)
    shapes_masks = ~(range_row < path_lengths[:, None])  # shape (B, T), True where not padding
    return padded_shapes, shapes_masks
                       
