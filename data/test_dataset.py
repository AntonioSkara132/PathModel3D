from data_utils import ShapeDataset as Dataset
import matplotlib.pyplot as plt

dataset = Dataset("/home/antonio/ws/LangPath3D/data/blender_primitives_np", 32)

path = dataset[35]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_aspect('equal')

ax.scatter(path[:, 0], path[:, 1], path[:, 2])
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
plt.show()

