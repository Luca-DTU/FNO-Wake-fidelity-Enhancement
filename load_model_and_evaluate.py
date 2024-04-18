
import os
from neuralop.utils import count_model_params
import torch
from neuralop.datasets.tensor_dataset import TensorDataset
from src.data_scripts import data_loading
from omegaconf import OmegaConf
import pickle
from neuralop import LpLoss, H1Loss
from neuralop.models import TFNO
from src.utils import SuperResolutionTFNO
model_folder ="outputs/2024-04-15/15-22-31"
config_path = model_folder+"/.hydra/config.yaml"
model_path = os.path.join(model_folder,"model.pth")
data_processor_path = os.path.join(model_folder,"data_processor.pkl")

# read the config file
config = OmegaConf.load(config_path)
# load the test data
data_source = getattr(data_loading, config.data_source.name)()
### override
config.data_source.test_args.input_spacing = 2.0
config.data_source.test_args.output_spacing = 4.0
###
x_test,y_test = data_source.extract(**config.data_source.test_args)
# load model
if config.data_format.positional_encoding:
    input_channels = x_test.shape[1]+2
else:
    input_channels = x_test.shape[1]
out_channels = y_test.shape[1]
if config.super_resolution:
    model = SuperResolutionTFNO(**config.TFNO, in_channels=input_channels, 
                                out_channels=out_channels, out_size=y_test.shape[2:])
else:
    model = TFNO(**config.TFNO, in_channels=input_channels, out_channels=out_channels)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
# load data processor
with open(data_processor_path, "rb") as f:
    data_processor = pickle.load(f)
# eval losses
l2loss = LpLoss(d=2, p=2,reduce_dims=[0,1]) # d=2 is the spatial dimension, p=2 is the L2 norm, reduce_dims=[0,1] means that the loss is averaged over the spatial dimensions 0 and 1
h1loss = H1Loss(d=2,reduce_dims=[0,1]) # d=2 is the spatial dimension, reduce_dims=[0,1] means that the loss is averaged over the spatial dimensions 0 and 1
eval_losses = {'l2': l2loss, 'h1': h1loss}
test_batch_size = config.train.test_batch_size
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
data_processor = data_processor.to("cpu")
test_loss = data_source.evaluate(test_loader,model,data_processor,losses=eval_losses,save = False, **config.data_source.evaluate_args)
