from matplotlib import pyplot as plt
import numpy as np
import re
import os


# filepath = "multirun/2024-05-11/12-37-11/7/main.log"
filepath = "multirun/2024-05-23/12-16-54/0/main.log"
# read the file
with open(filepath, 'r') as f:
    lines = f.readlines()
# extract all lines like " [2024-05-11 19:39:52,101][__main__][INFO] - Epoch 499, train error: 0.1960142392378587" in a dict
pattern = re.compile(r"\[.*\]\[__main__\]\[INFO\] - Epoch (\d+), train error: (\d+\.\d+)")
data = {}
for line in lines:
    match = pattern.match(line)
    if match:
        epoch, error = match.groups()
        data[int(epoch)] = float(error)
# plot the data
epochs = np.array(list(data.keys()))
errors = np.array(list(data.values()))
# make log plot
plt.figure(figsize=(9, 3))
plt.semilogy(epochs, errors)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.grid()
plt.tight_layout()
plt.show()

