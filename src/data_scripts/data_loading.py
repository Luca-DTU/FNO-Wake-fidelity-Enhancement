import numpy as np
import xarray
import torch
import pickle
from matplotlib import pyplot as plt
import logging
import hydra
from tqdm import tqdm

log = logging.getLogger(__name__)

class DataExtractor():
    def __init__(self):
        pass
    def extract(self):
        raise NotImplementedError
    
    def plot_output(self,output, y, titles = ["U"], save = True):
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
            if save:
                plt.savefig(f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/output_{titles[i]}.png")
                plt.close()
            else:
                plt.show()

    def evaluate_sample(self,test_loader, model,data_processor,output_names,plot=True,**kwargs):
        test_samples = test_loader.dataset
        index = np.random.randint(0, len(test_samples))
        data = test_samples[index]
        data["x"] = data["x"].unsqueeze(0)
        data["y"] = data["y"].unsqueeze(0)
        data_processor.train = False 
        data = data_processor.preprocess(data, batched=True)
        output = model(data["x"])
        output, data = data_processor.postprocess(output, data)
        y = data["y"]
        # detach and convert to numpy
        output = output.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        # plot
        if plot:
            self.plot_output(output, y, titles = output_names, **kwargs)

    
    def evaluate(self,test_loader, model,data_processor,losses,**kwargs):
        self.evaluate_sample(test_loader, model,data_processor,**kwargs)
        test_loss = self.evaluate_all(test_loader, model,data_processor,losses)
        return test_loss


    def evaluate_all(self,test_loader, model,data_processor,losses):
        total_loss = {name: 0 for name in losses.keys()}
        for batch in test_loader:
            data_processor.train = False
            batch = data_processor.preprocess(batch, batched=True)
            output = model(batch["x"])
            output, batch = data_processor.postprocess(output, batch)
            y = batch["y"]
            # detach and convert to numpy
            output = output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            # compute loss
            loss = {}
            output = torch.tensor(output).float()
            y = torch.tensor(y).float()
            for name, loss_fn in losses.items():
                loss[name] = loss_fn(output, y).item()
                log.info(f"Test loss {name}: {loss[name]}")
                total_loss[name] += loss[name]
        log.info(f"Test loss: {total_loss}")
        return sum(total_loss.values())
class rans(DataExtractor):    
    def extract_sample(self,dataset, layout, wd, inputs, outputs):
        x = [dataset[inp].interp(type=layout, wd=wd).data for inp in inputs]
        x = np.stack(x,0)
        y = [dataset[out].interp(type=layout, wd=wd).data for out in outputs]
        y = np.stack(y,0)
        return x, y

    def extract(self,turbines_per_side = 4, 
                    horizontal_grid_spacing = [0.5,1.0,2.0,4.0],
                    layout_type = np.arange(0,50), 
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
            for layout in tqdm(layout_type):
                for wd in inflow_wind_direction:
                    x, y = self.extract_sample(dataset, layout, wd, inputs, outputs)
                    # print(x.shape, y.shape)
                    x_list.append(x)
                    y_list.append(y)
            # close dataset
            dataset.close()
        if len(horizontal_grid_spacing) > 1:
            size_single_spacing = len(x_list)//len(horizontal_grid_spacing)
            x_list = [np.stack(x_list[i*size_single_spacing:(i+1)*size_single_spacing],0) for i in range(len(horizontal_grid_spacing))]
            y_list = [np.stack(y_list[i*size_single_spacing:(i+1)*size_single_spacing],0) for i in range(len(horizontal_grid_spacing))]
            x_list = [torch.tensor(x).float() for x in x_list]
            y_list = [torch.tensor(y).float() for y in y_list]
            return x_list, y_list
        else:
            x_train = np.stack(x_list,0)
            y_train = np.stack(y_list,0)
            x_train = torch.tensor(x_train).float()
            y_train = torch.tensor(y_train).float()
            return x_train, y_train
    

class rans_prescaled_independently(rans):
    # can this be achieved with super
    def extract_sample(self,dataset, layout, wd, inputs, outputs):
        x, y = super().extract_sample(dataset, layout, wd, inputs, outputs)
        # min max scale y over the last two dimensions to allow for multivariate output
        y = (y-np.min(y, axis = (-2,-1), keepdims = True))/(np.max(y, axis = (-2,-1), keepdims = True)-np.min(y, axis = (-2,-1), keepdims = True))
        return x, y



class rans_prescaled_on_resolution(rans):
    def __init__(self):
        super().__init__()
        self.min = {}
        self.max = {}
    def min_max_scale(self,x,horizontal_grid_spacing = None):
        min = np.min(x, axis = (0,2,3), keepdims = True)
        max = np.max(x, axis = (0,2,3), keepdims = True)
        self.min[horizontal_grid_spacing] = min.flatten()
        self.max[horizontal_grid_spacing] = max.flatten()
        return (x-min)/(max-min)

    def extract(self,turbines_per_side = 4, 
                    horizontal_grid_spacing = [0.5,1.0,2.0,4.0],
                    layout_type = np.arange(0,50), 
                    inflow_wind_direction = [315.0],
                    path = 'data/RANS/',
                    inputs = ["A_AWF"],
                    outputs = ['U', 'V', 'W', 'P', 'muT', 'tke', 'epsilon']):
        x,y = super().extract(turbines_per_side = turbines_per_side,
                    horizontal_grid_spacing = horizontal_grid_spacing,
                    layout_type = layout_type,
                    inflow_wind_direction = inflow_wind_direction,
                    path = path,
                    inputs = inputs,
                    outputs = outputs)
        # min_max_scale = lambda x: (x-np.min(x, axis = (0,2,3), keepdims = True))/(np.max(x, axis = (0,2,3), keepdims = True)-np.min(x, axis = (0,2,3), keepdims = True))
        if len(horizontal_grid_spacing) > 1:
            y = [torch.tensor(self.min_max_scale(y_.numpy(),spacing)).float() for spacing,y_ in zip(horizontal_grid_spacing,y)]
        else:
            y = torch.tensor(self.min_max_scale(y.numpy())).float()
        return x, y
    
            

        
    
class synthetic_data(DataExtractor):
    def extract(self,resolution = 32,path = "data/simulated_data/synth_const_dil.pkl", multivariate = True, reduce = None):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(resolution, int):
            data = data[resolution]
            x_train = data["x"]
            if not multivariate:
                y_train = data["y"][:,1,:,:].reshape(-1,1,resolution,resolution)
            else:
                y_train = data["y"]
            if reduce is not None:
                x_train = x_train[:int(len(x_train)*reduce)]
                y_train = y_train[:int(len(y_train)*reduce)]
            x_train = torch.tensor(x_train).float()
            y_train = torch.tensor(y_train).float()
            return x_train, y_train
        elif hasattr(resolution, "__iter__"): # check if resolution is iterable
            x_list = []
            y_list = []
            for res in resolution:
                data_res = data[res]
                x_train = data_res["x"]
                if not multivariate:
                    y_train = data_res["y"][:,1,:,:].reshape(-1,1,res,res)
                else:
                    y_train = data_res["y"]
                if reduce is not None:
                    x_train = x_train[:int(len(x_train)*reduce)]
                    y_train = y_train[:int(len(y_train)*reduce)]
                x_train = torch.tensor(x_train).float()
                y_train = torch.tensor(y_train).float()
                x_list.append(x_train)
                y_list.append(y_train)
            return x_list, y_list

def investigate_ranges():
    # investigate the ranges of the different datasets
    # note: this is not in the requirements.txt
    from sklearn.linear_model import LinearRegression
    extractor = rans()
    inflow_wind_direction = [270.0, 273.0, 276.0, 279.0, 282.0, 285.0, 288.0, 291.0, 294.0, 297.0,
                            300.0, 303.0, 306.0, 309.0, 312.0, 315.0]
    x,y = extractor.extract(horizontal_grid_spacing = [0.5,1.0,2.0,4.0],
                          inflow_wind_direction = inflow_wind_direction, outputs = ['U'])
    # pick a sample and plot
    sample = 27
    fig, ax = plt.subplots(2,4,figsize=(10,5))
    # plot with colorbar for each axis
    for i in range(4):
        x_ = x[i][sample,0]
        y_ = y[i][sample,0]
        im = ax[0,i].imshow(x_)
        fig.colorbar(im, ax=ax[0,i], orientation='vertical')
        im = ax[1,i].imshow(y_)
        fig.colorbar(im, ax=ax[1,i], orientation='vertical')
        # remove ticks
        for a in ax[:,i]:
            a.set_xticks([])
            a.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"rans_resolution_comparison.png")
    plt.show()
    # show linear regression
    x_res = np.array([0.5,1.0,2.0,4.0]).reshape(-1,1)
    y_res = [np.min(y_.numpy()) for y_ in y]
    reg = LinearRegression().fit(x_res, y_res)
    print(reg.coef_, reg.intercept_)
    from sklearn.metrics import r2_score
    r2 = round(r2_score(y_res, reg.predict(x_res)),3)
    plt.figure(figsize=(5,5))
    plt.scatter(x_res, y_res)
    plt.plot(x_res, reg.predict(x_res))
    plt.xlabel("Horizontal grid spacing")
    plt.ylabel("Min U")
    # add text box in top-right corner with r2
    plt.text(4.12, 0.732, f"$R^2$: {r2}",
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.savefig("min_U_regression.png")
    plt.show()

class rans_super_resolution(rans):
    def extract_single_spacing(self,spacing):
        x_list = []
        databasename = 'awf_database_%gcD.nc' % spacing
        databasename = self.path + databasename
        dataset = xarray.open_dataset(databasename)
        for layout in tqdm(self.layout_type):
            for wd in self.inflow_wind_direction:
                _, y = self.extract_sample(dataset, layout, wd, self.inputs, self.outputs)
                x_list.append(y)
        # close dataset
        dataset.close()
        train_data = np.stack(x_list,0)
        train_data = torch.tensor(train_data).float()
        return train_data
    def extract(self, 
                input_spacing = 0.5,
                output_spacing = 1.0,
                layout_type = np.arange(0,50),
                inflow_wind_direction = [315.0],
                path = 'data/RANS/',
                inputs = ["A_AWF"],
                outputs = ['U', 'V', 'W', 'P', 'muT', 'tke', 'epsilon'],
                **kwargs):
        self.path = path
        self.inputs = inputs
        self.outputs = outputs
        self.layout_type = layout_type
        self.inflow_wind_direction = inflow_wind_direction
        x_train = self.extract_single_spacing(input_spacing)
        y_train = self.extract_single_spacing(output_spacing)
        return x_train, y_train
        
if __name__ == "__main__":
    investigate_ranges()


