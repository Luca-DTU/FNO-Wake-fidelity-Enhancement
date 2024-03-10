import numpy as np
import xarray
import torch
import pickle
from matplotlib import pyplot as plt
import logging
import hydra
log = logging.getLogger(__name__)

class DataExtractor():
    def __init__(self):
        pass
    def extract(self):
        raise NotImplementedError
    
    def plot_output(self,output, y, titles = ["U"]):
        for i in range(output.shape[1]): # output channels
            fig, ax = plt.subplots(1,3,figsize=(15,5))
            fig.suptitle(f"{titles[i]}")
            im = ax[0].imshow(output[0,i])
            fig.colorbar(im, ax=ax[0], orientation='horizontal')
            im = ax[1].imshow(y[0,i])
            fig.colorbar(im, ax=ax[1], orientation='horizontal')
            im = ax[2].imshow(output[0,i]-y[0,i])
            fig.colorbar(im, ax=ax[2], orientation='horizontal')
            ax[0].set_title("Predicted")
            ax[1].set_title("True")
            ax[2].set_title("Difference")
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])
            plt.tight_layout()
            plt.savefig(f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/output_{titles[i]}.png")
            plt.show()

    def evaluate(self,test_loader, model,data_processor,output_names,losses,plot=True):
        test_samples = test_loader.dataset
        index = np.random.randint(0, len(test_samples))
        data = test_samples[index]
        data_processor.train = False
        # forward pass
        data = data_processor.preprocess(data, batched=False)
        output = model(data["x"].unsqueeze(0))
        if data_processor.out_normalizer and not data_processor.train:
            output = data_processor.out_normalizer.inverse_transform(output)
        y = data["y"].unsqueeze(0)
        # detach and convert to numpy
        output = output.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        # plot
        if plot:
            self.plot_output(output, y, titles = output_names)
        # compute loss
        loss = {}
        output = torch.tensor(output).float()
        y = torch.tensor(y).float()
        for name, loss_fn in losses.items():
            loss[name] = loss_fn(output, y).item()
            log.info(f"Test loss {name}: {loss[name]}")
        return sum(loss.values())

class rans(DataExtractor):    
    def extract_sample(self,dataset, s, wd, inputs, outputs):
        x = [dataset[inp].interp(s=s, wd=wd).data for inp in inputs]
        x = np.stack(x,0)
        y = [dataset[out].interp(s=s, wd=wd).data for out in outputs]
        y = np.stack(y,0)
        return x, y

    def extract(self,turbines_per_side = 4, 
                    horizontal_grid_spacing = [0.5,1.0,2.0,4.0],
                    turbine_interspacing = [4.0,8.0], 
                    inflow_wind_direction = [315.0],
                    path = 'data/RANS/',
                    inputs = ["A_AWF"],
                    outputs = ['U', 'V', 'W', 'P', 'muT', 'tke', 'epsilon']):
        x_list = []
        y_list = []
        for horizontal_spacing in horizontal_grid_spacing:
            databasename = 'awf_database_%gcD.nc' % horizontal_spacing
            databasename = path + databasename
            dataset = xarray.open_dataset(databasename)
            for s in turbine_interspacing:
                for wd in inflow_wind_direction:
                    x, y = self.extract_sample(dataset, s, wd, inputs, outputs)
                    print(x.shape, y.shape)
                    x_list.append(x)
                    y_list.append(y)
            # close dataset
            dataset.close()
        x_train = np.stack(x_list,0)
        y_train = np.stack(y_list,0)
        x_train = torch.tensor(x_train).float()
        y_train = torch.tensor(y_train).float()
        return x_train, y_train
    
class synthetic_data(DataExtractor):
    def extract(self,resolution = 32,path = "data/simulated_data/synth_const_dil.pkl", multivariate = True):
        with open(path, "rb") as f:
            data = pickle.load(f)
        data = data[resolution]
        x_train = data["x"]
        if not multivariate:
            y_train = data["y"][:,1,:,:].reshape(-1,1,resolution,resolution)
        else:
            y_train = data["y"]
        x_train = torch.tensor(x_train).float()
        y_train = torch.tensor(y_train).float()
        return x_train, y_train