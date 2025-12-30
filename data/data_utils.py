from augmentation import rotate
import os
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

class ShapeDataset(Dataset):
    def __init__(self, directory, num_augment):
        self.directory = directory
        self.num_augment = num_augment
        self.names = os.listdir(directory)
        #self.names = [name.removesuffix(".npy") for name in self.filenames]

    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        file_idx = idx % self.num_augment
        shape = np.load(os.path.join(self.directory, self.names[file_idx]))
        shape = rotate(shape)
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
                       
