from src.data_scripts.data_loading import rans
from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
def investigate_angles(*args, **kwargs):
    # investigate the ranges of the different datasets
    # note: this is not in the requirements.txt
    extractor = rans()
    x,y = extractor.extract(*args,**kwargs)
    # pick a sample and plot
    fig, ax = plt.subplots(1,x.shape[0],figsize=(10,5))
    # plot with colorbar for each axis
    for i in range(x.shape[0]):
        x_ = x[i,0]
        y_ = y[i,0]
        im = ax[i].imshow(x_)
        ax[i].set_title(str(kwargs['inflow_wind_direction'][i]))
    # remove ticks
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.show()


def investigate_ranges(*args, **kwargs):
    # investigate the ranges of the different datasets
    # note: this is not in the requirements.txt
    
    extractor = rans()
    x,y = extractor.extract(*args,**kwargs)
    # pick a sample and plot
    sample = 27
    fig, ax = plt.subplots(2,4,figsize=(6,4))
    vmin_x = min(torch.min(x_[sample,0]) for x_ in x).item()
    vmax_x = max(torch.max(x_[sample,0]) for x_ in x).item()
    vmin_y = min(torch.min(y_[sample,0]) for y_ in y).item()
    vmax_y = max(torch.max(y_[sample,0]) for y_ in y).item()
    # plot with colorbar for each axis
    for i in range(4):
        x_ = x[i][sample,0]
        y_ = y[i][sample,0]
        im_x = ax[0,i].imshow(x_,vmin=vmin_x, vmax=vmax_x, cmap='viridis_r')
        im_y = ax[1,i].imshow(y_,vmin=vmin_y, vmax=vmax_y)
        # remove ticks
        for a in ax[:,i]:
            a.set_xticks([])
            a.set_yticks([])
    fig.colorbar(im_x, ax=ax[0,3], orientation='vertical')
    fig.colorbar(im_y, ax=ax[1,3], orientation='vertical')
    plt.tight_layout()
    plt.savefig(f"rans_resolution_comparison.png")
    plt.show()
    # show linear regression
    x_res = np.array([0.5,1.0,2.0,4.0]).reshape(-1,1)
    y_res = [np.min(y_.numpy()) for y_ in y]
    reg = LinearRegression().fit(x_res, y_res)
    print(reg.coef_, reg.intercept_)
    r2 = round(r2_score(y_res, reg.predict(x_res)),3)
    plt.figure(figsize=(5,4))
    plt.scatter(x_res, y_res)
    plt.plot(x_res, reg.predict(x_res))
    plt.xticks([0.5,1.0,2.0,4.0],["2D","D","D/2","D/4"])
    plt.xlabel("Horizontal grid spacing")
    plt.ylabel("Min U")
    # add text box in top-right corner with r2
    plt.text(4.12, 0.731, f"$R^2$: {r2}",
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.savefig("min_U_regression.png")
    plt.show()



if __name__ == "__main__":
    inflow_wind_direction = [270.0, 273.0, 276.0, 279.0, 282.0, 285.0, 288.0, 291.0, 294.0, 297.0,
                            300.0, 303.0, 306.0, 309.0, 312.0, 315.0]
    investigate_ranges(path = "data/RANS_Newton/",
                       horizontal_grid_spacing = [0.5,1.0,2.0,4.0],
                    inflow_wind_direction = inflow_wind_direction, outputs = ['U'],
                    inputs = ["A_AWF"])
    investigate_ranges(path = "data/RANS_Newton/",
                       horizontal_grid_spacing = [0.5,1.0,2.0,4.0],
                    inflow_wind_direction = inflow_wind_direction, outputs = ['U'],
                    inputs = ["Fx"])


    # inflow_wind_directions = [270.0, 275.0, 280.0, 285.0, 290.0, 295.0, 300.0, 305.0, 310.0, 315.0]
    inflow_wind_directions = [270.0, 280.0, 290.0, 300.0, 310.0]
    investigate_angles(path = "data/RANS_Newton/",
                    horizontal_grid_spacing = [2.0],
                    inflow_wind_direction = inflow_wind_directions, outputs = ['U'], layout_type = [28],
                    inputs = ["Fx"])
    


            
