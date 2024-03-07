
import torch
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop import LpLoss, H1Loss
from neuralop.training.callbacks import Callback
from src.utils import data_format
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from src.data_scripts import data_loading
import hydra
from omegaconf import OmegaConf
import logging
log = logging.getLogger(__name__)
import os 
import shutil


def main(config):
    data_source = getattr(data_loading, config.data_source.name)()
    x_train, y_train = data_source.extract(**config.data_source.train_args)
    x_test,y_test = data_source.extract(**config.data_source.test_args)
    train_loader, test_loader, data_processor = data_format(x_train,y_train,x_test,y_test,
                                                            batch_size = config.train.batch_size,
                                                            test_batch_size= config.train.batch_size,
                                                            encode_output=config.data_format.encode_output,
                                                            encode_input=config.data_format.encode_input,
                                                            positional_encoding=config.data_format.positional_encoding,
                                                            grid_boundaries=config.data_format.grid_boundaries,
                                                            )
    if config.data_format.positional_encoding:
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

    test_loss = data_source.evaluate(test_loader,model, data_processor,losses=eval_losses,**config.data_source.evaluate_args)
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
