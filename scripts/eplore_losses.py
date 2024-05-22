from neuralop.losses import LpLoss,H1Loss
from src.trainer import linear,axial_bump,center_sink,linear_legacy
from src.trainer import weightedLpLoss
import numpy as np
import torch
if __name__ == "__main__":
    yhat = np.random.rand(2,3,4,4)
    y = np.random.rand(2,3,4,4)
    yhat = torch.tensor(yhat,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    l2loss = LpLoss(d=2, p=2,reduce_dims=[0,1]) # d=2 is the spatial dimension, p=2 is the L2 norm, reduce_dims=[0,1] means that the loss is averaged over the spatial dimensions 0 and 1
    h1loss = H1Loss(d=2,reduce_dims=[0,1]) # d=2 is the spatial dimension, reduce_dims=[0,1] means that the loss is averaged over the spatial dimensions 0 and 1
    linear_loss = weightedLpLoss(weight_fun=[linear]) 
    axial_bump_loss = weightedLpLoss(weight_fun=[axial_bump])
    center_sink_loss = weightedLpLoss(weight_fun=[center_sink])
    linear_legacy_loss = weightedLpLoss(weight_fun=[linear_legacy])
    results = {}
    results['L2 Loss'] = l2loss(yhat,y)
    results['H1 Loss'] = h1loss(yhat,y)
    results['Linear Loss'] = linear_loss(yhat,y)
    results['Axial Bump Loss'] = axial_bump_loss(yhat,y)
    results['Center Sink Loss'] = center_sink_loss(yhat,y)
    results['Linear Legacy Loss'] = linear_legacy_loss(yhat,y)
    for key in results:
        print(f'{key}: {results[key]}')
