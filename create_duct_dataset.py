import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.interpolate import griddata
from tqdm import tqdm

def plot_comparison(y, z, ux, uy, uz, yi, zi, uix, uiy, uiz):
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    ax[0, 0].scatter(y, z, c=ux)
    ax[0, 0].set_title("Ux")
    ax[0, 1].scatter(y, z, c=uy)
    ax[0, 1].set_title("Uy")
    ax[0, 2].scatter(y, z, c=uz)
    ax[0, 2].set_title("Uz")
    ax[1, 0].scatter(yi, zi, c=uix)
    ax[1, 0].set_title("Ux")
    ax[1, 1].scatter(yi, zi, c=uiy)
    ax[1, 1].set_title("Uy")
    ax[1, 2].scatter(yi, zi, c=uiz)
    ax[1, 2].set_title("Uz")
    plt.show()

def load_data(base, geometry):
    cx = f"{base}{geometry}_Cx.npy"
    cy = f"{base}{geometry}_Cy.npy"
    cz = f"{base}{geometry}_Cz.npy"
    x = np.load(cx)
    y = np.load(cy)
    z = np.load(cz)
    ux = f"{base}{geometry}_Ux.npy"
    uy = f"{base}{geometry}_Uy.npy"
    uz = f"{base}{geometry}_Uz.npy"
    ux = np.load(ux)
    uy = np.load(uy)
    uz = np.load(uz)
    return x, y, z, ux, uy, uz

def uniform_grid(y, z, ux, uy, uz):
    # Define a target uniform grid
    yi = np.linspace(y.min(), y.max(), 100)  # Uniform grid y-coordinates
    zi = np.linspace(z.min(), z.max(), 100)  # Uniform grid z-coordinates
    yi, zi = np.meshgrid(yi, zi)  # Create a meshgrid for the uniform grid
    # Interpolate from the non-uniform grid to the uniform grid
    uix = griddata((y, z), ux, (yi, zi), method='linear')
    uiy = griddata((y, z), uy, (yi, zi), method='linear')
    uiz = griddata((y, z), uz, (yi, zi), method='linear')
    return yi, zi, uix, uiy, uiz

def save_data(y, z, ux, uy, uz,Re,res):
    np.save(f"data/ductDataset/{Re}_{res}_y.npy", y)
    np.save(f"data/ductDataset/{Re}_{res}_z.npy", z)
    np.save(f"data/ductDataset/{Re}_{res}_ux.npy", ux)
    np.save(f"data/ductDataset/{Re}_{res}_uy.npy", uy)
    np.save(f"data/ductDataset/{Re}_{res}_uz.npy", uz)

if __name__ == "__main__":
    base = "data/opendata/kepsilon"
    files = os.listdir(base)
    selected = [f for f in files if "DUCT" 
                in f and ("Ux" in f or "Uy" in f or "Uz" in f or
                "Cx" in f or "Cy" in f or "Cz" in f)] 
    reynolds = [f.split("_")[-2] for f in selected]
    reynolds = sorted(list(set(reynolds)))
    for Re in tqdm(reynolds):
        geometry = f"/kepsilon_DUCT_{Re}"
        x, y, z, ux, uy, uz = load_data(base, geometry)
        save_data(y, z, ux, uy, uz,Re,"Original")
        yi, zi, uix, uiy, uiz = uniform_grid(y, z, ux, uy, uz)
        save_data(yi, zi, uix, uiy, uiz,Re,"100")
        # downsample the uniform grid
        y_down = yi[::2, ::2]
        z_down = zi[::2, ::2]
        uix_down = uix[::2, ::2]
        uiy_down = uiy[::2, ::2]
        uiz_down = uiz[::2, ::2]
        save_data(y_down, z_down, uix_down, uiy_down, uiz_down,Re,"50")
        # downsample the uniform grid
        y_down = yi[::4, ::4]
        z_down = zi[::4, ::4]
        uix_down = uix[::4, ::4]
        uiy_down = uiy[::4, ::4]
        uiz_down = uiz[::4, ::4]
        save_data(y_down, z_down, uix_down, uiy_down, uiz_down,Re,"25")

        


    


