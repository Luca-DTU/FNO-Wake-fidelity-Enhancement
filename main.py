import numpy as np
from matplotlib import pyplot as plt
import torch
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop import LpLoss, H1Loss
from neuralop.training.callbacks import Callback
from src.utils import data_format
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import xarray
import hydra
from omegaconf import OmegaConf
import logging
log = logging.getLogger(__name__)
import os 
import shutil

def extract_sample(dataset, s, wd, inputs, outputs):
    x = [dataset[inp].interp(s=s, wd=wd).data for inp in inputs]
    x = np.stack(x,0)
    y = [dataset[out].interp(s=s, wd=wd).data for out in outputs]
    y = np.stack(y,0)
    return x, y

def extract_data(turbines_per_side = 4, 
                 hotizontal_grid_spacing = [0.5,1.0,2.0,4.0],
                 turbine_interspacing = [4.0,8.0], 
                 inflow_wind_direction = [315.0],
                 path = 'data/RANS/',
                 inputs = ["A_AWF"],
                 outputs = ['U', 'V', 'W', 'P', 'muT', 'tke', 'epsilon']):
    x_list = []
    y_list = []
    for horizontal_spacing in hotizontal_grid_spacing:
        databasename = 'awf_database_%gcD.nc' % horizontal_spacing
        databasename = path + databasename
        dataset = xarray.open_dataset(databasename)
        for s in turbine_interspacing:
            for wd in inflow_wind_direction:
                x, y = extract_sample(dataset, s, wd, inputs, outputs)
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

def plot_output(output, y, titles = ["U"]):
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
        plt.show()

def evaluate_model(test_loader, model,data_processor,output_names,losses,plot=True):
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
        plot_output(output, y, titles = output_names)
    # compute loss
    loss = {}
    output = torch.tensor(output).float()
    y = torch.tensor(y).float()
    for name, loss_fn in losses.items():
        loss[name] = loss_fn(output, y).item()
        log.info(f"Test loss {name}: {loss[name]}")
    return sum(loss.values())


def main(config):
    outputs = config.data.outputs
    x_train, y_train = extract_data(hotizontal_grid_spacing=[1.0],outputs=outputs)
    x_test,y_test = extract_data(hotizontal_grid_spacing=[2.0],outputs=outputs)
    train_loader, test_loader, data_processor = data_format(x_train,y_train,x_test,y_test,
                                                            batch_size = config.train.batch_size,
                                                            test_batch_size= config.train.batch_size,
                                                            encode_output=config.data.encode_output,
                                                            encode_input=config.data.encode_input,
                                                            positional_encoding=config.data.positional_encoding,
                                                            grid_boundaries=config.data.grid_boundaries,
                                                            )
    if config.data.positional_encoding:
        input_channels = x_train.shape[1]+2
    else:
        input_channels = x_train.shape[1]
    data_processor = data_processor.to(device)
    model = TFNO(n_modes=(config.TFNO.n_modes, config.TFNO.n_modes), 
                hidden_channels=config.TFNO.hidden_channels, 
                projection_channels=config.TFNO.projection_channels, 
                factorization=config.TFNO.factorization,
                rank=config.TFNO.rank,
                in_channels=input_channels, out_channels=y_train.shape[1])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=config.adam.lr, 
                                    weight_decay=config.adam.weight_decay)
    scheduler = getattr(torch.optim.lr_scheduler, config.scheduler.name)(optimizer, **config.scheduler.args)
    l2loss = LpLoss(d=2, p=2,reduce_dims=[0,1])
    h1loss = H1Loss(d=2,reduce_dims=[0,1])
    train_loss = h1loss
    eval_losses={'h1': h1loss, 'l2': l2loss}
    class LogLoss(Callback):
        def on_epoch_end(self,epoch, train_err, avg_loss):
            log.info(f"Epoch {epoch}, train error: {train_err}")

    trainer = Trainer(model=model, n_epochs=config.train.epochs,
                    device=device,
                    data_processor=data_processor,
                    wandb_log=False,
                    log_test_interval=1, # log at every epoch
                    use_distributed=False,
                    verbose=True,
                    callbacks=[LogLoss()]
                    )

    trainer.train(train_loader=train_loader,
                test_loaders={"test":test_loader},
                optimizer=optimizer,
                scheduler=scheduler, 
                regularizer=False, 
                training_loss=train_loss,
                eval_losses=eval_losses,
                )

    test_loss = evaluate_model(test_loader,model, data_processor, outputs, 
                               losses=eval_losses,plot=True)
    return test_loss


@hydra.main(config_path="conf", config_name="base",version_base=None)
def my_app(config):
    # Run the main function
    log.info(f"Running with config: {OmegaConf.to_yaml(config)}")
    # try:
    test_loss = main(config) # the main function should return the test loss to optimize the hyperparameters
    # except Exception as e:
    #     print("-----------------------------------")
    #     print("JOB FAILED --- EXCEPTION")
    #     log.error(f"Exception: {e}")
    #     print("CONFIGURATION")
    #     print(f"Running with config: {OmegaConf.to_yaml(config['training'])}")
    #     print("-----------------------------------")
    #     test_loss = 1e10
    return test_loss

def clean_up_empty_files():
    outputs_folder = "outputs"
    for root, dirs, files in os.walk(outputs_folder):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if os.path.isdir(dir_path):
                subdirs = os.listdir(dir_path)
                if len(subdirs) == 2 and "main.log" in subdirs and ".hydra" in subdirs:
                    shutil.rmtree(dir_path)

if __name__ == '__main__':
    clean_up_empty_files()
    my_app()
