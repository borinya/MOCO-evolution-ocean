import torch
import torch.nn as nn
from torchvision import models as torchvision_models
from moco.builder import MoCo_ResNet
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial

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
from datetime import datetime

import numpy as np
# import vits

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
    checkpoint = torch.load(checkpoint_path, map_location='cuda')  # Use 'cpu' or 'cuda' as needed

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

# features = model.base_encoder(input_tensor)

# Dummy input tensor
input_tensor = torch.randn(16, 7, 349, 661).cuda()  # Adjust size as needed

# Get features
features = get_features(model, input_tensor)
print(features.shape)  # Should output something like (1, 256)
print('zaebis')





    # for i, images in enumerate(train_loader):
        
    #     if args.gpu is not None:
    #         images[0] = images[0].cuda(args.gpu, non_blocking=True) 
    #         images[1] = images[1].cuda(args.gpu, non_blocking=True)

    #     # compute output
    #     loss = model(images[0], images[1], moco_m)
