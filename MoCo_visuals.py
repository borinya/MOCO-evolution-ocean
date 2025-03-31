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


import numpy as np



import torch
from torchviz import make_dot

# Assuming you have the model created like this:
model = moco.builder.create_moco_model(
    arch='resnet50',
    in_channels=7,
    dim=256,
    mlp_dim=4096,
    T=1.0
)

# Create a dummy input tensor. Adjust the dimensions accordingly.
# Assuming input is a batch of images with size [batch_size, channels, height, width]
batch_size = 2  # Example batch size
height, width = 224, 224  # Example height and width for ResNet
dummy_input1 = torch.randn(batch_size, 7, height, width)
dummy_input2 = torch.randn(batch_size, 7, height, width)
momentum_value = 0.999  # Example momentum value

# Forward pass through the model to create the computation graph
output = model(dummy_input1, dummy_input2, momentum_value)

# Generate the visualization
dot = make_dot(output, params=dict(list(model.named_parameters())))

# Save the graph to a file (optional) or display it
dot.render("moco_model_graph", format="png")  # This will save the graph as a PNG file
dot.view()  # This will open the rendered graph in your default image viewer

# import vits