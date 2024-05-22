from neuralop.datasets.output_encoder import UnitGaussianNormalizer
from neuralop.datasets.tensor_dataset import TensorDataset
from neuralop.datasets.transforms import PositionalEmbedding2D, Transform
from neuralop.datasets.data_transforms import DefaultDataProcessor as parentProcessor
import torch
from torch.utils.data.dataset import Dataset

class DefaultDataProcessor(parentProcessor):

    def preprocess(self, data_dict, batched=True):
        x = data_dict['x'].to(self.device)
        y = data_dict['y'].to(self.device)

        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x, batched=batched)
        if self.out_normalizer is not None and self.train:
            y = self.out_normalizer.transform(y)

        data_dict['x'] = x
        data_dict['y'] = y

        return data_dict

    def postprocess(self, output, data_dict):
        y = data_dict['y']
        if self.out_normalizer and not self.train:
            output = self.out_normalizer.inverse_transform(output)
            # y = self.out_normalizer.inverse_transform(y)
        data_dict['y'] = y
        return output, data_dict

class MultiResolutionDataset(Dataset):
    def __init__(self, X_set, y_set, shuffle= False, mode = "batch_wise"):
        assert len(X_set) == len(y_set), "Size mismatch between tensors"
        self.n = len(X_set) # number of resolutions
        if self.n > 1:
            for j in range(1,self.n):
                assert X_set[j].size(0) == X_set[0].size(0), "Size mismatch between tensors"
                assert y_set[j].size(0) == y_set[0].size(0), "Size mismatch between tensors"
                if shuffle:
                    idx = torch.randperm(X_set[j].size(0))
                    X_set[j] = X_set[j][idx]
                    y_set[j] = y_set[j][idx]


        self.X_set = X_set
        self.y_set = y_set
        self.shuffle = shuffle
        self.mode = mode

    def __getitem__(self, index):
        if self.n > 1:
            items = []
            for j in range(self.n):
                items.append({'x': self.X_set[j][index], 'y': self.y_set[j][index]})
        return items

    def __len__(self):
        return self.X_set[0].size(0)


def data_format_multi_resolution(x_train:torch.tensor,y_train:torch.tensor,x_test:torch.tensor,y_test:torch.tensor,
                encoding:str = "channel-wise",
                encode_input:bool = False ,encode_output:bool = False,
                grid_boundaries:tuple = [[-0.5, 0.5], [-0.5, 0.5]],
                batch_size:int = 4,
                test_batch_size:int = 4,
                positional_encoding:bool = True,
                use_rans_encoder:bool = True,
                multi_res_kwargs:dict = {}
                ):

    if encode_input:
        if encoding == "channel-wise":
            reduce_dims = list(range(x_train[0].ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]
        input_encoders = [UnitGaussianNormalizer(dim=reduce_dims) for _ in x_train]
        for i in range(len(x_train)):
            input_encoders[i].fit(x_train[i])
    else:
        input_encoders = [None for _ in x_train]

    if encode_output:
        if use_rans_encoder:
            output_encoders = [rans_custom_encoder() for _ in y_train]
        else:
            if encoding == "channel-wise":
                reduce_dims = list(range(y_train[0].ndim))
            elif encoding == "pixel-wise":
                reduce_dims = [0]
            output_encoders = [UnitGaussianNormalizer(dim=reduce_dims) for _ in y_train]
            for i in range(len(y_train)):
                output_encoders[i].fit(y_train[i])
    else:
        output_encoders = [None for _ in y_train]

    train_db = MultiResolutionDataset(
        x_train,
        y_train,
        **multi_res_kwargs
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
    data_processors = [DefaultDataProcessor(
        in_normalizer=input_encoders[i],
        out_normalizer=output_encoders[i],
        positional_encoding=pos_encoding
    ) for i in range(len(x_train))]

    return train_loader, test_loader, data_processors

class rans_custom_encoder(Transform):
    """Custom encoder for RANS data
    transform the output to 1-output to have the data close to zero instead of close to 1
    """
    def __init__(self):
        super().__init__()
    def transform(self,x):
        return 1-x
    def inverse_transform(self,x):
        return 1-x
    def forward(self,x):
        return self.transform(x)
    def to(self,device):
        return self
    
    
def data_format(x_train:torch.tensor,y_train:torch.tensor,x_test:torch.tensor,y_test:torch.tensor,
                encoding:str = "channel-wise",
                encode_input:bool = False ,encode_output:bool = False,
                grid_boundaries:tuple = [[-0.5, 0.5], [-0.5, 0.5]],
                batch_size:int = 4,
                test_batch_size:int = 4,
                positional_encoding:bool = True,
                use_rans_encoder:bool = True
                ):
    if encode_input:
        if encoding == "channel-wise":
            reduce_dims = list(range(x_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]
        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        input_encoder.fit(x_train)
    else:
        input_encoder = None

    if encode_output:
        if use_rans_encoder:
            output_encoder = rans_custom_encoder()
        else:
            if encoding == "channel-wise":
                reduce_dims = list(range(y_train.ndim))
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            output_encoder.fit(y_train)
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


import torch.nn as nn
from neuralop.models import TFNO
from neuralop.layers.mlp import MLP

class SuperResolutionProjection(MLP):
    def __init__(self, out_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assume the base MLP class outputs (batch size, channels out, *size)
        # We need to add a final layer to map to the final resolution, out_size.
        # Using an upsampling layer. Adjust this to match your specific needs (e.g., interpolation method).
        self.upsample = nn.Upsample(size=out_size, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # First, pass through the original MLP layers
        x = super().forward(x)
        # Then upsample to the desired output size
        x = self.upsample(x)
        return x

class SuperResolutionTFNO(TFNO):
    def __init__(self, out_size: tuple,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection = SuperResolutionProjection(out_size,
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=self.non_linearity
        )
