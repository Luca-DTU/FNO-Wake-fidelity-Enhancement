from src.data_scripts import data_loading
from matplotlib import pyplot as plt
import numpy as np
data_source = data_loading.rans()
all_directions = np.arange(270, 318, 3)
all_layouts = np.arange(50)
horizontal_grid_spacing = [1.0]

# plot all
x_num = 5
y_num = 10
n_figures = len(all_directions)
for i,direction in enumerate(all_directions):
    x,y = data_source.extract(inflow_wind_direction=[direction], 
                    layout_type=all_layouts, 
                    horizontal_grid_spacing=horizontal_grid_spacing,
                    outputs=["U"])
    fig, axs = plt.subplots(x_num,y_num, figsize=(10,10))
    for j in range(x_num):
        for k in range(y_num):
            index = j*y_num+k
            axs[j,k].imshow(x[index,0,:,:])
            axs[j,k].set_xticks([])
            axs[j,k].set_yticks([])
    plt.tight_layout()
    plt.savefig(f"latex_figures/layouts/{direction}.png")
    plt.close()