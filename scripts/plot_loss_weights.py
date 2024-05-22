import torch
from src.trainer import linear,axial_bump,center_sink
from matplotlib import pyplot as plt
diff = torch.zeros(233,109)
weightings = [linear,axial_bump,center_sink]
weights = [weight_fun(diff) for weight_fun in weightings]
# pick a sample and plot
fig, ax = plt.subplots(2,len(weightings)+1,figsize=(6,6))
# plot with colorbar for each axis
for i,weight in enumerate(weights):
    im = ax[0,i].imshow(weight,vmin=0, vmax=1)
    # ax[i].set_title(weight_fun.__name__)
# perumations
im = ax[1,0].imshow(weights[0]*weights[1],vmin=0, vmax=1)
im = ax[1,1].imshow(weights[0]*weights[2],vmin=0, vmax=1)
im = ax[1,2].imshow(weights[1]*weights[2],vmin=0, vmax=1)
im = ax[1,3].imshow(weights[0]*weights[1]*weights[2],vmin=0, vmax=1)

# Remove the last subplot axis if not used
ax[0, -1].remove()

# remove ticks
for a in ax.flatten():
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
cbar_ax = fig.add_axes([0.75, 0.51, 0.03, 0.465])  # Modify these values as needed
fig.colorbar(im, cax=cbar_ax)
plt.show()