from neuralop.datasets.output_encoder import UnitGaussianNormalizer
from neuralop.datasets.tensor_dataset import TensorDataset
from neuralop.datasets.transforms import PositionalEmbedding2D
from neuralop.datasets.data_transforms import DefaultDataProcessor
import torch

def data_format(x_train:torch.tensor,y_train:torch.tensor,x_test:torch.tensor,y_test:torch.tensor,
                encoding:str = "channel-wise",
                encode_input:bool = False ,encode_output:bool = False,
                grid_boundaries:tuple = [[-0.5, 0.5], [-0.5, 0.5]],
                batch_size:int = 4,
                test_batch_size:int = 4,
                positional_encoding:bool = True
                ):
    
    if encode_input:
        if encoding == "channel-wise":
            reduce_dims = list(range(x_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        input_encoder.fit(x_train)
        #x_train = input_encoder.transform(x_train)
        #x_test = input_encoder.transform(x_test.contiguous())
    else:
        input_encoder = None

    if encode_output:
        if encoding == "channel-wise":
            reduce_dims = list(range(y_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        output_encoder.fit(y_train)
        #y_train = output_encoder.transform(y_train)
    else:
        output_encoder = None

    train_db = TensorDataset(
        x_train,
        y_train,
    )
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

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
    
    if positional_encoding:
        pos_encoding = PositionalEmbedding2D(grid_boundaries=grid_boundaries)
    else:
        pos_encoding = None
    data_processor = DefaultDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder,
        positional_encoding=pos_encoding
    )
    return train_loader, test_loader, data_processor