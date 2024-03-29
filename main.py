import numpy as np
import torch
from neuralop.models import TFNO
# from neuralop import Trainer
from src.trainer import Trainer, MultiResTrainer
from neuralop import LpLoss, H1Loss
from neuralop.training.callbacks import Callback
from src.utils import data_format, data_format_multi_resolution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from src.data_scripts import data_loading
import hydra
from omegaconf import OmegaConf
import logging
log = logging.getLogger(__name__)
import os 
import shutil
torch.manual_seed(42)
from neuralop.utils import count_model_params

def main(config):
    data_source = getattr(data_loading, config.data_source.name)()
    x_train, y_train = data_source.extract(**config.data_source.train_args)
    x_test,y_test = data_source.extract(**config.data_source.test_args)
    def model_setup(config,input_channels,out_channels):
        model = TFNO(**config.TFNO,
                     in_channels=input_channels, out_channels=out_channels)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=config.adam.lr, 
                                        weight_decay=config.adam.weight_decay)
        scheduler = getattr(torch.optim.lr_scheduler, config.scheduler.name)(optimizer, **config.scheduler.args)
        l2loss = LpLoss(d=2, p=2,reduce_dims=[0,1]) # d=2 is the spatial dimension, p=2 is the L2 norm, reduce_dims=[0,1] means that the loss is averaged over the spatial dimensions 0 and 1
        h1loss = H1Loss(d=2,reduce_dims=[0,1]) # d=2 is the spatial dimension, reduce_dims=[0,1] means that the loss is averaged over the spatial dimensions 0 and 1
        train_loss = h1loss
        eval_losses={'h1': h1loss, 'l2': l2loss}
        return model, optimizer, scheduler, train_loss, eval_losses
    class LogLoss(Callback):
        def on_epoch_end(self,epoch, train_err, avg_loss):
            log.info(f"Epoch {epoch}, train error: {train_err}")

    if config.multi_resolution:
        train_loader, test_loader, data_processors = data_format_multi_resolution(x_train,y_train,x_test,y_test,
                                                            batch_size = config.train.batch_size,
                                                            test_batch_size= config.train.test_batch_size,
                                                            encode_output=config.data_format.encode_output,
                                                            encode_input=config.data_format.encode_input,
                                                            positional_encoding=config.data_format.positional_encoding,
                                                            grid_boundaries=config.data_format.grid_boundaries,
                                                            )
        if config.data_format.positional_encoding:
            input_channels = x_train[0].shape[1]+2
        else:
            input_channels = x_train[0].shape[1]
        data_processors = [data_processor.to(device) for data_processor in data_processors]
        out_channels = y_train[0].shape[1]
        model, optimizer, scheduler, train_loss, eval_losses = model_setup(config,input_channels,out_channels)
        log.info(f'\nOur model has {count_model_params(model)} parameters.')
        trainer = MultiResTrainer(model=model, n_epochs=config.train.epochs,
                        device=device,
                        data_processors=data_processors,
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
        # store model
        torch.save(model.state_dict(), f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/model.pth")
        # evaluate
        test_loss = data_source.evaluate(test_loader,model, data_processors[-1],losses=eval_losses,**config.data_source.evaluate_args)
    else:
        train_loader, test_loader, data_processor = data_format(x_train,y_train,x_test,y_test,
                                                                batch_size = config.train.batch_size,
                                                                test_batch_size= config.train.test_batch_size,
                                                                encode_output=config.data_format.encode_output,
                                                                encode_input=config.data_format.encode_input,
                                                                positional_encoding=config.data_format.positional_encoding,
                                                                grid_boundaries=config.data_format.grid_boundaries,
                                                                )
    
        if config.data_format.positional_encoding:
            input_channels = x_train.shape[1]+2
        else:
            input_channels = x_train.shape[1]
        out_channels = y_train.shape[1]
        data_processor = data_processor.to(device)
        model, optimizer, scheduler, train_loss, eval_losses = model_setup(config,input_channels,out_channels)
        log.info(f'\nOur model has {count_model_params(model)} parameters.')
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
        # store model
        torch.save(model.state_dict(), f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/model.pth")
        # evaluate
        test_loss = data_source.evaluate(test_loader,model,data_processor,losses=eval_losses,**config.data_source.evaluate_args)
    return test_loss


@hydra.main(config_path="conf", config_name="synth_data_sweeper",version_base=None)
def my_app(config):
    # Run the main function
    log.info(f"Running with config: {OmegaConf.to_yaml(config)}")
    try:
        test_loss = main(config) # the main function should return the test loss to optimize the hyperparameters
        if test_loss is None or np.isnan(test_loss):
            raise ValueError("Test loss is None")
    except Exception as e:
        print("-----------------------------------")
        print("JOB FAILED --- EXCEPTION")
        log.error(f"Exception: {e}")
        print("CONFIGURATION")
        print(f"Running with config: {OmegaConf.to_yaml(config)}")
        print("-----------------------------------")
        test_loss = 1e10
    return test_loss

def clean_up_empty_files(outputs_folder = "outputs"):
    for root, dirs, files in os.walk(outputs_folder):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if os.path.isdir(dir_path):
                subdirs = os.listdir(dir_path)
                if len(subdirs) == 0:
                    shutil.rmtree(dir_path)
                else:
                    for subdir in subdirs:
                        subdir_path = os.path.join(dir_path,subdir)
                        if os.path.isdir(subdir_path):
                            subsubdirs = os.listdir(subdir_path)
                            if len(subsubdirs) <= 2 and "main.log" in subsubdirs and ".hydra" in subsubdirs:
                                shutil.rmtree(subdir_path)

if __name__ == '__main__':
    clean_up_empty_files(outputs_folder="outputs")
    clean_up_empty_files(outputs_folder="multirun")
    my_app()
