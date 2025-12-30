import random

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

data = np.load("augmented_cone.npy")
print(data.shape)

idx = random.randint(0, 100)

data = data[idx, :, :]
ax.scatter(data[:, 0], data[:, 1], data[:, 2])

plt.show()

