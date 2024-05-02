
import os
import torch
from neuralop.datasets.tensor_dataset import TensorDataset
from src.data_scripts import data_loading
from omegaconf import OmegaConf
import pickle
from neuralop import LpLoss, H1Loss
from neuralop.models import TFNO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils import SuperResolutionTFNO

def plot_wind_direction_error(series):
    # Extract the wind directions and MSE values from the series
    wind_directions = series.index.values
    mse_values = series.values
    # Convert degrees to radians
    radians = np.deg2rad(wind_directions)
    # Create a polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 4))
    ax.plot(radians, mse_values, 'o-', linewidth=1.5)
    ax.set_theta_zero_location('N')  # Set 0 degrees to the top (North)
    ax.set_theta_direction(-1)  # Clockwise
    # Display only the specified slice
    ax.set_thetamin(min(wind_directions))  # Minimum boundary for theta
    ax.set_thetamax(max(wind_directions))  # Maximum boundary for theta
    plt.tight_layout()
    plt.savefig("wind_direction_error.png")
    plt.show()


model_folder = "multirun/2024-04-27/15-16-35/8" # baest multi-res
config_path = model_folder+"/.hydra/config.yaml"
model_path = os.path.join(model_folder,"model.pth")
data_processor_path = os.path.join(model_folder,"data_processors.pkl")
# read the config file
config = OmegaConf.load(config_path)
# load the test data
data_source = getattr(data_loading, config.data_source.name)()
all_directions = config.data_source.test_args.inflow_wind_direction
loss_dict = {}
for direction in all_directions:
    config.data_source.test_args.inflow_wind_direction = [direction]
    test_args = config.data_source.train_args
    test_args.update(config.data_source.test_args)
    x_test,y_test = data_source.extract(**test_args)
    # load model
    if "model" not in locals():
        if config.data_format.positional_encoding:
            input_channels = x_test.shape[1]+2
        else:
            input_channels = x_test.shape[1]
        out_channels = y_test.shape[1]

        if "non_linearity" in config.TFNO:
            non_linearity = getattr(torch.nn.functional, config.TFNO.non_linearity) 
            kwargs = OmegaConf.to_container(config.TFNO)
            kwargs["non_linearity"] = non_linearity
        else:
            kwargs = OmegaConf.to_container(config.TFNO)

        if config.get("super_resolution"):
            model = SuperResolutionTFNO(**kwargs, in_channels=input_channels, 
                                        out_channels=out_channels, out_size=y_test.shape[2:])
        else:
            model = TFNO(**kwargs, in_channels=input_channels, out_channels=out_channels)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        # load data processor
        with open(data_processor_path, "rb") as f:
            data_processor = pickle.load(f)
        if isinstance(data_processor, list):
            data_processor = data_processor[-1]
        data_processor = data_processor.to("cpu")
        # eval losses
        l2loss = LpLoss(d=2, p=2,reduce_dims=[0,1]) # d=2 is the spatial dimension, p=2 is the L2 norm, reduce_dims=[0,1] means that the loss is averaged over the spatial dimensions 0 and 1
        h1loss = H1Loss(d=2,reduce_dims=[0,1]) # d=2 is the spatial dimension, reduce_dims=[0,1] means that the loss is averaged over the spatial dimensions 0 and 1
        eval_losses = {'l2': l2loss, 'h1': h1loss}
        # test_batch_size = config.train.test_batch_size
        test_batch_size = 4
    # dummy test loader
    test_db = TensorDataset(
        x_test,
        y_test,
    )
    test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loss = data_source.evaluate(test_loader,model,data_processor,losses=eval_losses,save = False, plot=False, output_names = ["U"])
    loss_dict[direction] = test_loss
loss_df = pd.Series(loss_dict)
plot_wind_direction_error(loss_df)