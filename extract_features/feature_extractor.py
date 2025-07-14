import os
import torch
import numpy as np
from extract_features.config import CHECKPOINT_PATH, MOCO_DIM

def replace_batchnorm_with_identity(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.BatchNorm2d):
            setattr(model, name, torch.nn.Identity())
        else:
            replace_batchnorm_with_identity(module)

def initialize_model():
    from moco.builder import MoCo_ResNet
    import torchvision.models as torchvision_models
    import torch.nn as nn
    from functools import partial

    model = MoCo_ResNet(
        partial(torchvision_models.__dict__['resnet50'], zero_init_residual=True), 
        MOCO_DIM, 4096, 1.0
    )

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

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    model = initialize_model()
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint.get('epoch', 0)
    return model, epoch

def get_features(model, input_tensor):
    model.eval()
    with torch.no_grad():
        features = model.base_encoder(input_tensor)
        return features