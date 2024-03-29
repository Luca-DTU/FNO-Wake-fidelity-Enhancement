
import os
from neuralop.utils import count_model_params
import torch
from src.data_scripts import data_loading
from omegaconf import OmegaConf
model_folder ="multirun/2024-03-11/20-31-05/44"
config_path = model_folder+"/.hydra/config.yaml"
model_path = os.path.join(model_folder,"model.pth")

# read the config file
config = OmegaConf.load(config_path)
# load the test data
data_source = getattr(data_loading, config.data_source.name)()
x_test,y_test = data_source.extract(**config.data_source.test_args)