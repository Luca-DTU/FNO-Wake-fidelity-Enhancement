# train at 50x50, predict at 100x100
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from src.utils import data_format
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop import LpLoss, H1Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_data(resolution = 50, multivariate = False):
    path = "data/ductDataset/"
    files = os.listdir(path)
    files = [f for f in files if f.split("_")[1] == str(resolution)]
    Re_uniques = sorted(list(set([f.split("_")[0] for f in files])))

    binary_image = np.zeros((resolution, resolution), dtype=int)
    binary_image[0, :] = 1  # Top boundary
    binary_image[-1, :] = 1  # Bottom boundary
    binary_image[:, 0] = 1  # Left boundary
    binary_image[:, -1] = 1  # Right boundary
    if multivariate:
        y_train = np.zeros((len(Re_uniques),3,resolution,resolution))
    else:
        y_train = np.zeros((len(Re_uniques),1,resolution,resolution))
    x_train = np.zeros((len(Re_uniques),2,resolution,resolution))
    for ii,Re in enumerate(Re_uniques):
        ux = np.load(f"{path}{Re}_{resolution}_ux.npy")
        uy = np.load(f"{path}{Re}_{resolution}_uy.npy")
        uz = np.load(f"{path}{Re}_{resolution}_uz.npy")
        all_data = np.stack([ux, uy, uz], axis=0)
        if multivariate:
            y_train[ii] = all_data
        else:
            y_train[ii] = ux.reshape(1,resolution,resolution)
        x_train[ii] = np.stack([binary_image, np.ones((resolution,resolution),dtype = float)*float(Re)], axis=0)
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    return x_train, y_train

if __name__ == "__main__":
    multivariate = True
    if multivariate:
        d = 3
    else:
        d = 2
    x_train, y_train = extract_data(resolution = 50, multivariate = multivariate)
    x_test,y_test = extract_data(resolution = 100, multivariate = multivariate)
    train_loader, test_loader, data_processor = data_format(x_train,y_train,x_test,y_test)
    test_loaders = {100: test_loader}
    # the positional encoding will add two positional channels to the input 
    # so if the input starts with two channels, the input will have 4 channels after the positional encoding
    data_processor = data_processor.to(device)

    # We create a tensorized FNO model
    model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42,
                in_channels=x_train.shape[1]+2, out_channels=y_train.shape[1])
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=1e-4, 
                                    weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    l2loss = LpLoss(d=d, p=2)
    h1loss = H1Loss(d=d)

    train_loss = h1loss
    eval_losses={'h1': h1loss, 'l2': l2loss}

    print('\n### MODEL ###\n', model)
    print('\n### OPTIMIZER ###\n', optimizer)
    print('\n### SCHEDULER ###\n', scheduler)
    print('\n### LOSSES ###')
    print(f'\n * Train: {train_loss}')
    print(f'\n * Test: {eval_losses}')

    trainer = Trainer(model=model, n_epochs=1000,
                    device=device,
                    data_processor=data_processor,
                    wandb_log=False,
                    log_test_interval=3,
                    use_distributed=False,
                    verbose=True)

    trainer.train(train_loader=train_loader,
                test_loaders=test_loaders,
                optimizer=optimizer,
                scheduler=scheduler, 
                regularizer=False, 
                training_loss=train_loss,
                eval_losses=eval_losses)


    test_samples = test_loaders[100].dataset

    
    index = np.random.randint(0, len(test_samples))
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y'].unsqueeze(0)
    # Model prediction
    out = model(x.unsqueeze(0))

    titles = ["U","U_pred","U_diff"]
    if multivariate:
        coordinates = ["x","y","z"] 
    else:
        coordinates = ["x"]
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