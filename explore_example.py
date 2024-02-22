from pathlib import Path
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

data_path = "C:/Users/lucab/miniconda3/envs/thesis/Lib/site-packages/neuralop/datasets/data"
train_resolution = 16  #32
channel_dim = 1
n_train = 1000

data = torch.load(
    Path(data_path).joinpath(f"darcy_train_{train_resolution}.pt").as_posix()
)
x_train = (
    data["x"][0:n_train, :, :].unsqueeze(channel_dim).type(torch.float32).clone()
)
y_train = data["y"][0:n_train, :, :].unsqueeze(channel_dim).clone()

print(x_train.shape, y_train.shape)

x = data["x"]
y = data["y"]

sample = np.random.randint(0, x.shape[0])
x_example = x[sample, :, :]
y_example = y[sample, :, :]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(x_example, cmap="viridis")
ax[0].set_title("Input")
ax[1].imshow(y_example, cmap="viridis")
ax[1].set_title("Output")
plt.show()

# plot animation of both x, y next to each other for comparison
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ims = []
for i in range(y.shape[0]):
    im1 = axs[0].imshow(x[i, :, :], cmap="viridis")
    im2 = axs[1].imshow(y[i, :, :], cmap="viridis")
    ims.append([im1, im2])
ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=1000)
plt.show()
