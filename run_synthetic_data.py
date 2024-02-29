
import numpy as np
from matplotlib import pyplot as plt
import torch
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop import LpLoss, H1Loss
import pickle
from neuralop.training.callbacks import Callback
from run_duct_data import data_format
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_data(resolution = 32):
    path = "data/simulated_data/synth_const_dil.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    data = data[resolution]
    x_train = data["x"]
    # dilation = data["dilation"]
    # # add the dilation as a channel to the input
    # output_array = np.zeros((dilation.shape[0], 1, resolution, resolution))
    # for i, value in enumerate(dilation):
    #     output_array[i] = np.full((1, resolution, resolution), value)
    # x_train = np.concatenate((x_train, output_array), axis=1)
    y_train = data["y"]
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    return x_train, y_train

if __name__ == "__main__":
    x_train, y_train = extract_data(resolution = 32)
    x_test,y_test = extract_data(resolution = 64)
    train_loader, test_loader, data_processor = data_format(x_train,y_train,x_test,y_test,
                                                            grid_boundaries=[[0, 1], [0, 1]],
                                                            batch_size = 64,
                                                            test_batch_size= 64
                                                            )
    test_loaders = {128: test_loader}
    # the positional encoding will add two positional channels to the input 
    # so if the input starts with two channels, the input will have 4 channels after the positional encoding
    data_processor = data_processor.to(device)

    # We create a tensorized FNO model
    model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42,
                in_channels=x_train.shape[1]+2, out_channels=y_train.shape[1])
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=1e-2, 
                                    weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    l2loss = LpLoss(d=2, p=2,reduce_dims=[0,1])
    h1loss = H1Loss(d=2,reduce_dims=[0,1])

    train_loss = h1loss
    eval_losses={'h1': h1loss, 'l2': l2loss}

    print('\n### MODEL ###\n', model)
    print('\n### OPTIMIZER ###\n', optimizer)
    print('\n### SCHEDULER ###\n', scheduler)
    print('\n### LOSSES ###')
    print(f'\n * Train: {train_loss}')
    print(f'\n * Test: {eval_losses}')

    class LogLoss(Callback):
        def on_epoch_end(self,epoch, train_err, avg_loss):
            print(f"Epoch {epoch}, train error: {train_err}, avg loss: {avg_loss}")

    trainer = Trainer(model=model, n_epochs=200,
                    device=device,
                    data_processor=data_processor,
                    wandb_log=False,
                    log_test_interval=1,
                    use_distributed=False,
                    verbose=True,
                    callbacks=[LogLoss()]
                    )

    trainer.train(train_loader=train_loader,
                test_loaders=test_loaders,
                optimizer=optimizer,
                scheduler=scheduler, 
                regularizer=False, 
                training_loss=train_loss,
                eval_losses=eval_losses,
                )


    test_samples = test_loaders[128].dataset
    index = np.random.randint(0, len(test_samples))
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Model prediction
    out = model(x.unsqueeze(0))
    # from generate_synthetic_data import min_max_normalize
    # out = min_max_normalize(out.detach().cpu().numpy())
    # out = torch.tensor(out).float()
    # Ground-truth
    # out, data = data_processor.postprocess(out, data)
    y = data['y'].unsqueeze(0)

    titles = ["U","U_pred","U_diff"]
    coordinates = ["x","y","z"] 
    for i in range(len(coordinates)):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Velocity component {coordinates[i]}")
        for j in range(3):
            title = titles[j]
            # If the title ends with "pred", use the 'out' variable, otherwise use 'y'
            data = out if title.endswith("pred") else y
            data = data.detach().cpu().numpy()
            # If the title ends with "diff", subtract the 'out' from 'y'
            if title.endswith("diff"):
                data = y - out.detach().cpu().numpy()
            # Display the image
            im = axs[j].imshow(data[0, i], cmap="viridis")
            # Remove the ticks
            axs[j].set_xticks([])
            axs[j].set_yticks([])
            # Set the title
            axs[j].set_title(f"{title}")
            fig.colorbar(im, ax=axs[j], orientation='horizontal')
        # Add a colorbar
        # fig.colorbar(im, ax=axs.ravel().tolist())
        plt.tight_layout()
        plt.show()