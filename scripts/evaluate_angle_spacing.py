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
    mean_error = np.mean(mse_values)
    # linspace with 1000 points for a smooth curve
    theta_all = np.linspace(min(radians), max(radians), 1000)
    # repeat the mean error for all points
    mse_all = np.repeat(mean_error, 1000)
    # Create a polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 4))
    ax.plot(radians, mse_values, '-', linewidth=1.5, label='L2 test loss')
    ax.plot(theta_all, mse_all, 'r--', linewidth=1.5, label='Mean L2 test loss')
    ax.set_theta_zero_location('N')  # Set 0 degrees to the top (North)
    ax.set_theta_direction(-1)  # Clockwise
    # Display only the specified slice
    # ax.set_thetamin(min(wind_directions))  # Minimum boundary for theta
    # ax.set_thetamax(max(wind_directions))  # Maximum boundary for theta
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.savefig("wind_direction_error.png")
    plt.show()

def evaluate_angle_spacing(model_folder = "multirun/2024-04-27/15-16-35/8"):  
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
        layout_loss_dict = {}
        for layout_type in range(50):
            config.data_source.test_args.layout_type = [layout_type]
        
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
                test_batch_size = 1
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
            layout_loss_dict[layout_type] = test_loss
        loss_dict[direction] = layout_loss_dict
    loss_df = pd.DataFrame(loss_dict)
    loss_df.index.name = "layout_type"
    return loss_df

if __name__ == "__main__":  
    model_folder = "multirun/2024-05-11/12-37-11/7"
    loss_df = evaluate_angle_spacing(model_folder=model_folder)
    # store the loss values in the model folder
    loss_df.to_csv(os.path.join(model_folder,"wind_direction_error.csv"))
    # loss_df = pd.read_csv(os.path.join(model_folder,"wind_direction_error.csv"),index_col=0)
    wind_dir_error = loss_df.mean(axis=0)
    # make wind error full circle based on symmetry
    # first mirror symmetry from 270-315 to 315-360
    index = np.arange(315,363,3)
    values = wind_dir_error.values[::-1]
    wind_dir_error_all = pd.concat([wind_dir_error,pd.Series(values,index=index)])
    # delete duplicated values
    wind_dir_error_all = wind_dir_error_all[~wind_dir_error_all.index.duplicated(keep='first')].iloc[:-1]
    # then repeat four times to cover the full circle
    wind_dir_error_all = pd.concat([wind_dir_error_all]*4)
    wind_dir_error_all.index = np.arange(0,360,3)
    plot_wind_direction_error(wind_dir_error_all)

    layout_error = loss_df.mean(axis=1)
    spacing_error = layout_error.groupby(layout_error.index // 10).mean()
    spacings = ["4D","5D","6D","7D","8D"]
    mean_spacing_error = spacing_error.mean()
    # Mean over each unit value
    shape_error = layout_error.groupby(layout_error.index % 10).mean()
    shape_error.plot(figsize=(5, 3), marker='o', label='L2 test loss')
    mean_shape_error = shape_error.mean()

    # plot everything in one plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    spacing_error.plot(ax=ax[0], marker='o', label='L2 test loss')
    ax[0].axhline(mean_spacing_error, color='r', linestyle='--', linewidth=1.5, label='Mean L2 test loss')
    ax[0].set_xlabel("Spacing")
    ax[0].set_xticks(range(len(spacings)))
    ax[0].set_xticklabels(spacings, rotation=0)
    ax[0].legend()
    shape_error.plot(ax=ax[1], marker='o', label='L2 test loss')
    ax[1].axhline(mean_shape_error, color='r', linestyle='--', linewidth=1.5, label='Mean L2 test loss')
    ax[1].set_xlabel("Shape")
    ax[1].set_xticks(range(10))
    ax[1].set_xticklabels(range(10), rotation=0)
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # plot everything in one plot including the wind direction error
    fig, ax = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    spacing_error.plot(ax=ax[0], marker='o', label='L2 test loss')
    ax[0].axhline(mean_spacing_error, color='r', linestyle='--', linewidth=1.5, label='Mean L2 test loss')
    ax[0].set_xlabel("Spacing")
    ax[0].set_xticks(range(len(spacings)))
    ax[0].set_xticklabels(spacings, rotation=0)
    # ax[0].legend()
    shape_error.plot(ax=ax[1], marker='o', label='L2 test loss')
    ax[1].axhline(mean_shape_error, color='r', linestyle='--', linewidth=1.5, label='Mean L2 test loss')
    ax[1].set_xlabel("Shape")
    ax[1].set_xticks(range(10))
    ax[1].set_xticklabels(range(10), rotation=0)
    # ax[1].legend()
    wind_dir_error.plot(ax=ax[2], marker='o', label='L2 test loss')
    mean_wind_dir_error = wind_dir_error.mean()
    ax[2].axhline(mean_wind_dir_error, color='r', linestyle='--', linewidth=1.5, label='Mean L2 test loss')
    ax[2].set_xlabel("Wind direction [Deg]")
    ax[2].legend(loc="best")
    plt.tight_layout()
    plt.show()

    # best performance
    idx,col = np.unravel_index(np.argmin(loss_df.values, axis=None), loss_df.values.shape)
    best_layout = loss_df.index[idx]
    best_wind_dir = loss_df.columns[col]
    print(f"Best layout: {best_layout}, Best wind direction: {best_wind_dir}")
    # worst performance
    idx,col = np.unravel_index(np.argmax(loss_df.values, axis=None), loss_df.values.shape)
    worst_layout = loss_df.index[idx]
    worst_wind_dir = loss_df.columns[col]
    print(f"Worst layout: {worst_layout}, Worst wind direction: {worst_wind_dir}")
