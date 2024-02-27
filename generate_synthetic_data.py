import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle

def generate_input_data(N, n_points, plot = False):
    base = np.zeros((N, N))
    x = np.random.randint(0, N, n_points)
    y = np.random.randint(0, N, n_points)
    base[x, y] = 1
    if plot:
        plt.imshow(base)
        plt.show()
    return base

def true_operator(input_data,dilation= 10, plot = False):
    N = input_data.shape[0]
    x, y = np.where(input_data != 0)
    n_points = len(x)
    output = np.zeros((3, N, N))
    for i in range(n_points):
        for j in range(N):
            for k in range(N):
                dist = np.sqrt((x[i] - j)**2 + (y[i] - k)**2)
                dist = dist/N # important, for resolution-invariance
                output[0, j, k] += np.exp(-dist / dilation)
                output[1, j, k] += np.sin(dist / dilation)
                output[2, j, k] += np.cos(dist / dilation)
    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(output[0])
        ax[0].set_title("Ux")
        ax[1].imshow(output[1])
        ax[1].set_title("Uy")
        ax[2].imshow(output[2])
        ax[2].set_title("Uz")
        plt.show()
    return output

def generate_dataset(N: list = [32,64],n_samples: list = [10,10]):
    dataset = {}
    for i,n in enumerate(N):
        data = np.zeros((n_samples[i],1,n, n))
        labels = np.zeros((n_samples[i],3,n, n))
        dilation = np.zeros((n_samples[i]))
        for j in tqdm(range(n_samples[i])):
            n_points = np.random.randint(1, np.ceil(n/10))
            dilation[j] = np.max((round(np.random.beta(1,2),3),0.001))
            input = generate_input_data(n, n_points)
            output = true_operator(input, dilation=dilation[j])
            data[j] = input
            labels[j] = output
        dataset[n] = {"x": data, "y": labels, "dilation": dilation}
    return dataset


if __name__ == "__main__":
    N = 128
    n_points = np.random.randint(1, np.ceil(N/10))
    dilation = round(np.random.beta(1,2),3)
    print(f"Number of samples: {n_points}, Dilation: {dilation}")
    input = generate_input_data(N, n_points, plot=True)
    output = true_operator(input, dilation=dilation, plot=True)
    simple_dataset = generate_dataset(N=[32,64,128],n_samples=[2000,1000,100])
    with open('data/simulated_data/synth_1.pkl', 'wb') as f: 
        pickle.dump(simple_dataset, f)
