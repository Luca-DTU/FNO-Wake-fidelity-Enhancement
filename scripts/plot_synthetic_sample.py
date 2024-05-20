from src.data_scripts.data_loading import synthetic_data
import matplotlib.pyplot as plt
if __name__ == '__main__':
    dataset = synthetic_data()
    x,y = dataset.extract(resolution = 64, reduce = 0.05)
    fig,ax = plt.subplots(2,3,figsize = (7,5))
    im = ax[0,1].imshow(x[0,0],vmin=-1, vmax=1)
    ax[0,0].remove()
    ax[0,2].remove()
    for i in range(3):
        ax[1,i].imshow(y[0,i],vmin=-1, vmax=1)
    for a in ax.flatten():
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    cbar_ax = fig.add_axes([0.75, 0.53, 0.03, 0.41])  # Modify these values as needed
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    