import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.evaluate_model import load_model
import numpy as np
from torch import nn

def visualize_path_3d(start_pt, out, shape, batch_idx=0, steps=None):
    """
    out:   [B, T, 3]  (velocities)
    shape: [B, N, 3]  (point cloud)
    """
    out = out.detach().cpu()
    out = torch.cumsum(out, 1) + start_pt
    shape_b = shape.detach().cpu()

    if steps is not None:
        out = out[:, :steps]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    shape_b = shape.squeeze(0).detach().numpy()
    # plot shape
    ax.scatter(
        shape_b[:, 0],
        shape_b[:, 1],
        shape_b[:, 2],
        s=3,
        alpha=0.3,
        label="Shape"
    )
    path_b = out.squeeze(0).detach().numpy()
    print(path_b)
    print(path_b.shape)

    # plot path
    ax.plot(
        path_b[:, 0],
        path_b[:, 1],
        path_b[:, 2],
        linewidth=2,
        color="red",
        label="Path"
    )

    # start / end markers
    ax.scatter(*path_b[0], color="green", s=60, label="Start")
    ax.scatter(*path_b[-1], color="black", s=60, label="End")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("3D Path Prediction")

    plt.tight_layout()
    plt.show()

import torch
shape = np.load("data/primitives/cube.npy")
shape = torch.from_numpy(shape)
shape_mask = torch.zeros([1, 512])
shape = shape.unsqueeze(0)

model = load_model("model1.pth", torch.device('cpu'))
tgt_len = 100
print(shape.shape)
print(shape_mask.shape)

start_pt = shape[0, 0, :]
start_pt = start_pt.unsqueeze(0).unsqueeze(0)
print(start_pt)

out, occupancy = model(start_pt, shape, shape_mask, tgt_len)
print(out)
out = out
visualize_path_3d(start_pt, out, shape, batch_idx=0, steps=100)


