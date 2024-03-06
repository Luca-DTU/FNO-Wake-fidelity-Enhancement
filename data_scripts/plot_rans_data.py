import numpy as np
import xarray
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')
# Latex font
plt.rcParams['font.family'] = 'STIXGeneral'
plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
plt.rcParams["legend.scatterpoints"] = 1
plt.rcParams["legend.numpoints"] = 1


def make_plot(databasename, s, wd, fmt='pdf'):
    dataset = xarray.open_dataset(databasename)
    flowvarnames = list(dataset.keys())
    flowvarnames.remove('wt_x')
    flowvarnames.remove('wt_y')
    nvar = len(flowvarnames)
    print('plot:', flowvarnames)
    cmap = plt.cm.get_cmap('jet')
    figx = 10.0
    figy = 3.0 * nvar
    fig, ax = plt.subplots(nvar, 1, sharey=True, sharex=True, figsize=(figx, figy))
    Xi, Yi = np.meshgrid(dataset['x'], dataset['y'])
    for i in range(len(flowvarnames)):
        # Make contour plot
        var = dataset[flowvarnames[i]].interp(s=s, wd=wd)
        vmin = var.min()
        vmax = var.max()
        if vmin == vmax:
            vmin = vmax * 0.99
        levels = np.linspace(vmin, vmax, 15)
        cs = ax[i].contourf(Xi, Yi, var.T, levels, cmap=cmap, vmax=vmax, vmin=vmin)  # plot flow
        ax[i].scatter(dataset['wt_x'].interp(s=s, wd=wd), dataset['wt_y'].interp(s=s, wd=wd)) # plot turbines
        ax[i].set_ylim(ax[i].get_ylim()[::-1])
        divider1 = make_axes_locatable(ax[i])
        cax1 = divider1.append_axes("right", size="2.5%", pad=0.05)
        cbar1 = plt.colorbar(cs, cax=cax1)
        ax[i].set_ylabel('$y/D$')
        ax[i].set_title(flowvarnames[i] + ' [' + dataset.variables[flowvarnames[i]].attrs['units'] + ']')
        ax[i].set_aspect('equal')
    ax[-1].set_xlabel('$x/D$')
    filename = 'ContourPlot.' + fmt
    fig.savefig(filename, dpi=600)


if __name__ == '__main__':
    n = 4  # Number turbines per side
    delta_x = 4.0  # Horizontal grid spacing [0.5, 1, 2, 4] in terms of cell per rotor diameter. For example, 4 means a spacing of D/4.
    databasename = 'awf_database_%gcD.nc' % delta_x
    path = 'data/RANS/'
    databasename = path + databasename
    spacing = 8.0  # Turbine inter spacing [4.0, 8.0] to be plotted
    wd = 315.0  # Inflow wind direction to be plotted (270-315 with 5 deg interval)
    make_plot(databasename, spacing, wd)
