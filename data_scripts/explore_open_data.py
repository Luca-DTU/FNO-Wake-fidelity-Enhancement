import numpy as np
import matplotlib.pyplot as plt
# https://www.nature.com/articles/s41597-021-01034-2
def plot_mesh(base, geometry):
    cx = f"{base}_{geometry}_Cx.npy"
    cy = f"{base}_{geometry}_Cy.npy"
    cz = f"{base}_{geometry}_Cz.npy"
    x = np.load(cx)
    y = np.load(cy)
    z = np.load(cz)
    print(x.shape, y.shape, z.shape)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(x, y)
    ax[0].set_title("x,y")
    ax[1].scatter(x, z)
    ax[1].set_title("x,z")
    ax[2].scatter(y, z)
    ax[2].set_title("y,z")
    plt.show()
    return x,y,z

def plot_speeds(base, geometry,x,y):
    ux = f"{base}_{geometry}_Ux.npy"
    uy = f"{base}_{geometry}_Uy.npy"
    uz = f"{base}_{geometry}_Uz.npy"
    ux = np.load(ux)
    uy = np.load(uy)
    uz = np.load(uz)
    print(ux.shape, uy.shape, uz.shape)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # contour plot of ux and uy on the x,y plane
    ax[0].scatter(x, y, c=ux)
    ax[0].set_title("Ux")
    ax[1].scatter(x, y, c=uy)
    ax[1].set_title("Uy")
    ax[2].scatter(x, y, c=uz)
    ax[2].set_title("Uz")
    plt.savefig(f"Figures/{geometry}_speeds.png")
    plt.show()

if __name__ == "__main__":
    base = "data/opendata/kepsilon/kepsilon"
    # geometry = "BUMP_h20" # h20 is the height of the bump
    # x,y,z = plot_mesh(base, geometry)
    # plot_speeds(base, geometry, x, y)
    geometry = "DUCT_1100" # THE NUMBERS ARE THE REYNOLDS NUMBER
    x,y,z = plot_mesh(base, geometry)
    plot_speeds(base, geometry, y, z)
    geometry = "DUCT_2600"
    plot_speeds(base, geometry, y, z)
    geometry = "DUCT_3500"
    plot_speeds(base, geometry, y, z)

