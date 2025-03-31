import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial
import pandas as pd
from moco.builder import MoCo_ResNet

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

import moco.builder
import moco.loader
import moco.optimizer
import moco.GLORYS12_dataset
from tqdm  import tqdm

import numpy as np


# print('zaebis загрузили модель')


def replace_batchnorm_with_identity(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.BatchNorm2d):
            setattr(model, name, torch.nn.Identity())
        else:
            replace_batchnorm_with_identity(module)



# Initialize the model architecture with the same parameters
def initialize_model():
    model = MoCo_ResNet(
        partial(torchvision_models.__dict__['resnet50'], zero_init_residual=True), 
        256, 4096, 1.0
    )
    
    # Modify the first convolutional layers if necessary
    input_channels = 7
    out_channels = model.base_encoder.conv1.out_channels
    kernel_size = model.base_encoder.conv1.kernel_size
    stride = model.base_encoder.conv1.stride
    padding = model.base_encoder.conv1.padding

    model.base_encoder.conv1 = nn.Conv2d(
        input_channels, out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    )
    
    model.momentum_encoder.conv1 = nn.Conv2d(
        input_channels, out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    )
    replace_batchnorm_with_identity(model)
    return model

# Load the model from a checkpoint
def load_checkpoint(checkpoint_path, optimizer=None, scaler=None):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')  # Use 'cpu' or 'cuda' as needed

    # Initialize the model
    model = initialize_model()
    
    # Load the model state dict
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    
    epoch = checkpoint.get('epoch', 0)
    
    return model, epoch

def get_features(model, input_tensor):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Forward pass to get features
        features = model.base_encoder(input_tensor)
        # Assuming the desired features are the output before the last layer
        # Modify this as per your model's architecture
        return features

checkpoint_path = '/app/MoCo/MOCOv3-MNIST/checkpoints/20250216_141630_checkpoint_0202.pth.tar'
model, start_epoch = load_checkpoint(checkpoint_path)

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


batch_size = 16
means = np.array([1.673302181686475, 33.37522164335293, 32.58433311325712, 
        11.152242330669477, 0.025353081653846376, -0.00907171541589713, 
        0.07366986763832623])

square_means = np.array([5.995956099912317, 1720.1733657260818, 1063.4138676153, 
        149.60278359811343, 0.009805976106874816, 0.008356788111581723, 
        0.035208209865639856])
stds = means**2 - square_means

augmentation2 = [
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
]
augmentation1 = [
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
]

train_dataset = moco.GLORYS12_dataset.Glorys12Dataset(csv_file='/app/MoCo/MOCOv3-MNIST/momental files and code/test_file_pathes_dataset.csv',
                        transform1=transforms.Compose(augmentation1),
                        transform2=transforms.Compose(augmentation2))   #moco.loader.TwoCropsTransform(transforms.Compose(augmentation2))                                                                                            transforms.Compose(augmentation2)))
test_loader = torch.utils.data.DataLoader(
train_dataset, batch_size=batch_size, shuffle=None,
num_workers=20, pin_memory=True, drop_last=False)



# Ensure the output directory exists
output_dir = '/app/LSTM_salmon/ocean_vectors'
os.makedirs(output_dir, exist_ok=True)

# Iterate over DataLoader
for i, images in enumerate(test_loader):

    # images.float().to(device)
    model.to(device)
    # images = [img.float() for img in images] 
    images[0] = images[0].cuda(0, non_blocking=True).float()


    # Extract features
    features = get_features(model, images[0])

    # Iterate over each item in the batch
    for batch_idx in range(features.size(0)):
        # Retrieve the index of the current item
        # dataset_idx = indices[batch_idx].item()

        # Retrieve the corresponding date
        idx_dataset = i*batch_size + batch_idx
        datetime_value = train_dataset.data_frame.loc[idx_dataset, 'Datetime']
        date_str = str(datetime_value)

        # Save the features to disk
        feature_vector = features[batch_idx].cpu().numpy()
        # filename = f"{date_str.replace(' ', '_').replace(':', '-')}_features.npy"
        filename = f"{idx_dataset}_{date_str.replace(' ', '_').replace(':', '-')}_features.npy"

        filepath = os.path.join(output_dir, filename)
        np.save(filepath, feature_vector)
        

    print(f"Processed batch {i} and saved feature vectors.")

print("All batches processed and feature vectors saved.")


