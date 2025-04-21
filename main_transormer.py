#!/usr/bin/env python
import argparse
import os
import torch
import torch.optim as optim 
import torchvision.transforms as transforms

import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from moco.transformer_model import OceanTransformer
from moco.transformer_dataset import Glorys12SequenceDataset
from moco.builder import MoCo_ResNet
from torchvision.models import resnet50
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(description='Ocean State Transformer Forecasting')
    
    # Paths
    parser.add_argument('--csv-file', required=False, type=str, default='/app/MoCo/MOCOv3-MNIST/momental files and code/cleaned_data.csv', 
                      help='Path to cleaned data CSV')
    parser.add_argument('--checkpoint', required=False, type=str, default='/app/MoCo/MOCOv3-MNIST/checkpoints/20250404_124558_checkpoint_0299.pth.tar',
                      help='Path to pretrained MoCo checkpoint')
    
    # Model architecture
    parser.add_argument('--transformer-layers', type=int, default=4,
                      help='Number of transformer layers')
    parser.add_argument('--transformer-heads', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--transformer-dim-ff', type=int, default=1024,
                      help='Feedforward dimension')
    parser.add_argument('--transformer-dropout', type=float, default=0.1,
                      help='Transformer dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Base learning rate')
    parser.add_argument('--encoder-lr', type=float, default=1e-5,
                      help='Encoder learning rate if finetuning')
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--optimizer', choices=['adam', 'adamw'], default='adamw')
    parser.add_argument('--finetune-encoder', action='store_true',
                      help='Fine-tune the encoder')
    
    # Data parameters
    parser.add_argument('--seq-len', type=int, default=60,
                      help='Input sequence length')
    parser.add_argument('--pred-horizon', type=int, default=30,
                      help='Prediction horizon')
    parser.add_argument('--predict-differences', action='store_true',
                      help='Predict differences instead of absolute values')
    parser.add_argument('--transform', type=str, default=None,
                      help='Type of transformation to apply')
    parser.add_argument('--cache-size', type=int, default=512,
                      help='Dataset cache size')
    parser.add_argument('--num-io-workers', type=int, default=20,
                      help='Number of IO workers for dataset preprocessing')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                      help='Prefetch factor for data loading')
    parser.add_argument('--random-seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=0,
                      help='Number of workers for DataLoader')
    
    # Experiment management
    parser.add_argument('--amp', action='store_true',
                      help='Use mixed precision')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                      help='Gradient clipping norm')
    parser.add_argument('--log-dir', type=str, default='/app/MoCo/MOCOv3-MNIST/runs_transformer',
                      help='Base directory for logs')
    parser.add_argument('--save-interval', type=int, default=10,
                      help='Checkpoint saving interval')
    
    return parser.parse_args()

def setup_logging(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"transformer_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    # Save all arguments to tensorboard
    for arg in vars(args):
        writer.add_text(f"args/{arg}", str(getattr(args, arg)))
        
    return writer, log_dir

def load_encoder(checkpoint_path, device, finetune=False):
    encoder = MoCo_ResNet(
        partial(resnet50, zero_init_residual=True), 
        dim=256, mlp_dim=4096, T=1.0
    ).base_encoder
    
    # Добавить weights_only=False для подавления предупреждения
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
    encoder.load_state_dict(state_dict, strict=False)
    
    if not finetune:
        for param in encoder.parameters():
            param.requires_grad = False
            
    return encoder.to(device)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logging and experiment tracking
    writer, log_dir = setup_logging(args)
    print(f"Experiment logs saved to: {log_dir}")
    
    # Initialize models
    encoder = load_encoder(args.checkpoint, device, args.finetune_encoder)

    transformer = OceanTransformer(
        input_dim=256,
        num_layers=args.transformer_layers,
        nhead=args.transformer_heads,
        dim_feedforward=args.transformer_dim_ff,
        dropout=args.transformer_dropout
    ).to(device)
    
    # Optimizer setup
    optim_params = [
        {'params': transformer.parameters(), 'lr': args.lr}
    ]
    if args.finetune_encoder:
        optim_params.append({'params': encoder.parameters(), 'lr': args.encoder_lr})
    
    optimizer = optim.AdamW(optim_params, weight_decay=args.weight_decay) if args.optimizer == 'adamw' \
        else optim.Adam(optim_params, weight_decay=args.weight_decay)
    
    # нормализация данных
    means = np.array([1.673302181686475, 33.37522164335293, 32.58433311325712, 
            11.152242330669477, 0.025353081653846376, -0.00907171541589713, 
            0.07366986763832623])
    
    square_means = np.array([5.995956099912317, 1720.1733657260818, 1063.4138676153, 
            149.60278359811343, 0.009805976106874816, 0.008356788111581723, 
            0.035208209865639856])
    stds = means**2 - square_means

    augmentation =  transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
])

    # Dataset and loader
    dataset = Glorys12SequenceDataset(
        csv_file=args.csv_file,
        sequence_length=args.seq_len,
        prediction_horizon=args.pred_horizon,
        predict_differences=args.predict_differences,
        transform=augmentation,
        cache_size=args.cache_size,
        num_io_workers=args.num_io_workers,
        prefetch_factor=args.prefetch_factor,
        random_seed=args.random_seed
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Training setup
    criterion = torch.nn.MSELoss()
    scaler = GradScaler(enabled=args.amp)
    global_step = 0
    
    for epoch in range(args.epochs):
        transformer.train()
        if args.finetune_encoder:
            encoder.train()
        else:
            encoder.eval()
            
        epoch_loss = 0.0
        
        for batch_idx, (sequences, targets) in enumerate(dataloader):
            sequences = sequences.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass through encoder
            with torch.set_grad_enabled(args.finetune_encoder):
                if args.amp:
                    with autocast():
                        batch_size, seq_len, H, W, C = sequences.shape
                        features = encoder(sequences.view(-1, H, W, C).view(batch_size, seq_len, -1))
                else:
                    batch_size, seq_len, H, W, C = sequences.shape
                    features = encoder(sequences.view(-1, H, W, C)).view(batch_size, seq_len, -1)
            
            # Transformer forward
            optimizer.zero_grad()
            
            if args.amp:
                with autocast():
                    predictions = transformer(features)
                    loss = criterion(predictions, targets)
            else:
                predictions = transformer(features)
                loss = criterion(predictions, targets)
            
            # Backward and optimize
            scaler.scale(loss).backward()
            
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.grad_clip)
                if args.finetune_encoder:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            epoch_loss += loss.item()
            global_step += 1
            writer.add_scalar('train/loss_step', loss.item(), global_step)
            
        # Epoch logging
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('train/loss_epoch', avg_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch+1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch+1,
                'transformer': transformer.state_dict(),
                'encoder': encoder.state_dict() if args.finetune_encoder else None,
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
                'loss': avg_loss
            }
            save_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    writer.close()

if __name__ == "__main__":
    main()

# #!/usr/bin/env python

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from datetime import datetime
# from torch.cuda.amp import GradScaler, autocast
# from netCDF4 import Dataset as netCDF4_Dataset
# import numpy as np
# import pandas as pd
# from functools import partial

# # Ваши кастомные модули
# from moco.transformer_dataset import Glorys12SequenceDataset
# from moco.builder import MoCo_ResNet
# from torchvision.models import resnet50

# class TransformerPredictor(nn.Module):
#     def __init__(self, input_dim=256, num_layers=4, nhead=8, dim_feedforward=1024):
#         super().__init__()
#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_dim, 
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#         self.output_proj = nn.Linear(input_dim, input_dim)
        
#     def forward(self, x):
#         # x: [batch_size, seq_len, features]
#         x = self.transformer(x)
#         x = self.output_proj(x[:, -1, :])  # Берем последний элемент последовательности
#         return x

# def load_encoder(checkpoint_path, device):
#     # Создаем модель MoCo и загружаем веса
#     encoder = MoCo_ResNet(
#         partial(resnet50, zero_init_residual=True), 
#         dim=256, mlp_dim=4096, T=1.0
#     )
    
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
#     encoder.load_state_dict(state_dict, strict=False)
    
#     # Берем только base_encoder и замораживаем веса
#     encoder = encoder.base_encoder
#     for param in encoder.parameters():
#         param.requires_grad = False
    
#     return encoder.to(device)

# def main():
#     # Конфигурация
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     checkpoint_path = "/app/MoCo/MOCOv3-MNIST/checkpoints/20250404_124558_checkpoint_0299.pth.tar"
#     csv_file = "/app/MoCo/MOCOv3-MNIST/momental files and code/cleaned_data.csv"
    
#     # Параметры обучения
#     batch_size = 32
#     seq_length = 60
#     pred_horizon = 30
#     num_epochs = 100
#     lr = 3e-4
    
#     # Инициализация моделей
#     encoder = load_encoder(checkpoint_path, device)
#     transformer = TransformerPredictor().to(device)
    
#     # Датасет и загрузчик
#     dataset = Glorys12SequenceDataset(
#         csv_file=csv_file,
#         sequence_length=seq_length,
#         prediction_horizon=pred_horizon
#     )
    
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # Оптимизатор и функция потерь
#     optimizer = optim.AdamW(transformer.parameters(), lr=lr)
#     criterion = nn.MSELoss()
#     scaler = GradScaler()
    
#     # Цикл обучения
#     for epoch in range(num_epochs):
#         transformer.train()
#         total_loss = 0.0
        
#         for sequences, targets in dataloader:
#             sequences = sequences.to(device)
#             targets = targets.to(device)
            
#             # Извлечение фичей
#             with torch.no_grad(), autocast():
#                 batch_size, seq_len, H, W, C = sequences.shape
#                 sequences = sequences.view(-1, H, W, C)  # [batch*seq_len, H, W, C]
#                 features = encoder(sequences.permute(0, 3, 1, 2))  # [batch*seq_len, 256]
#                 features = features.view(batch_size, seq_len, -1)  # [batch, seq_len, 256]
            
#             # Обучение трансформера
#             optimizer.zero_grad()
            
#             with autocast():
#                 predictions = transformer(features)
#                 loss = criterion(predictions, targets)
            
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
            
#             total_loss += loss.item()
        
#         avg_loss = total_loss / len(dataloader)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
#         # Сохранение модели
#         if (epoch+1) % 10 == 0:
#             torch.save({
#                 'epoch': epoch+1,
#                 'transformer_state': transformer.state_dict(),
#                 'optimizer_state': optimizer.state_dict(),
#                 'loss': avg_loss,
#             }, f"transformer_epoch_{epoch+1}.pth")

# if __name__ == "__main__":
#     main()
    
    
    
    # #!/usr/bin/env python

# # Copyright (c) Facebook, Inc. and its affiliates.
# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
# import os
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# import argparse
# import builtins
# import math
# import os
# import random
# import shutil
# import time
# import warnings
# from functools import partial

# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim
# import torch.multiprocessing as mp
# import torch.utils.data
# import torch.utils.data.distributed
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as torchvision_models
# from torch.utils.tensorboard import SummaryWriter

# import moco.builder
# import moco.loader
# import moco.optimizer
# import moco.GLORYS12_dataset
# from datetime import datetime

# import numpy as np
# # import vits


# torchvision_model_names = sorted(name for name in torchvision_models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(torchvision_models.__dict__[name]))

# model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

# parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')
# # parser.add_argument('data', metavar='DIR',
# #                     help='path to dataset')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet50)')
 
# parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', # тут самописное параллельство
#                     help='number of data loading workers (default: 0)')
#     # Параметры данных
# parser.add_argument('--io-workers', default=128, type=int, help='параллельная загрузка файлов')
# parser.add_argument('--cache-size', default=128, type=int, help='размер кэша датасета') # размер датасета 10115

# parser.add_argument('--epochs', default=100, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=32, type=int,
#                     metavar='N',
#                     help='mini-batch size (default: 64), this is the total '
#                          'batch size of all GPUs on all nodes when '
#                          'using Data Parallel or Distributed Data Parallel')
# parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
#                     metavar='LR', help='initial (base) learning rate', dest='lr')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
#                     metavar='W', help='weight decay (default: 1e-6)',
#                     dest='weight_decay')
# parser.add_argument('-p', '--print-freq', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')
#     # Добавляем аргумент для CSV файла
# parser.add_argument('-c', '--csv-file', default='/app/MoCo/MOCOv3-MNIST/momental files and code/test_file_pathes_dataset.csv',
#                         type=str, metavar='FILE', help='path to the CSV file (default: /app/MoCo/MOCOv3-MNIST/momental files and code/test_file_pathes_dataset.csv)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
# parser.add_argument('--seed', default=None, type=int,
#                     help='seed for initializing training. ')
# parser.add_argument('--gpu', default=0, type=int,
#                     help='GPU id to use.')
# parser.add_argument('--multiprocessing-distributed', action='store_true',
#                     help='Use multi-processing distributed training to launch '
#                          'N processes per node, which has N GPUs. This is the '
#                          'fastest way to use PyTorch for either single node or '
#                          'multi node data parallel training')

# # moco specific configs:
# parser.add_argument('--moco-dim', default=256, type=int,
#                     help='feature dimension (default: 256)')
# parser.add_argument('--moco-mlp-dim', default=4096, type=int,
#                     help='hidden dimension in MLPs (default: 4096)')
# parser.add_argument('--moco-m', default=0.99, type=float,
#                     help='moco momentum of updating momentum encoder (default: 0.99)')
# parser.add_argument('--moco-m-cos', action='store_true',
#                     help='gradually increase moco momentum to 1 with a '
#                          'half-cycle cosine schedule')
# parser.add_argument('--moco-t', default=1.0, type=float,
#                     help='softmax temperature (default: 1.0)')

# # vit specific configs:
# parser.add_argument('--stop-grad-conv1', action='store_true',
#                     help='stop-grad after first conv, or patch embedding')

# # other upgrades
# parser.add_argument('--optimizer', default='lars', type=str,
#                     choices=['lars', 'adamw'],
#                     help='optimizer used (default: lars)')
# parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
#                     help='number of warmup epochs')
# parser.add_argument('--crop-min', default=0.08, type=float,
#                     help='minimum scale for random cropping (default: 0.08)')

# # geo_channels default 7))
# parser.add_argument('--geo-channels', default=7, type=int,
#                     help='nomber of input data channels (default: 7)')

# def replace_batchnorm_with_identity(model):
#     for name, module in model.named_children():
#         if isinstance(module, torch.nn.BatchNorm2d):
#             setattr(model, name, torch.nn.Identity())
#         else:
#             replace_batchnorm_with_identity(module)
            
            


# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)



# def main():
#     args = parser.parse_args()

#     if args.seed is not None:
#         random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         cudnn.deterministic = True
#         warnings.warn('You have chosen to seed training. '
#                       'This will turn on the CUDNN deterministic setting, '
#                       'which can slow down your training considerably! '
#                       'You may see unexpected behavior when restarting '
#                       'from checkpoints.')

#     if args.gpu is not None:
#         warnings.warn('You have chosen a specific GPU. This will completely '
#                       'disable data parallelism.')

#     if args.dist_url == "env://" and args.world_size == -1:
#         args.world_size = int(os.environ["WORLD_SIZE"])

#     args.distributed = args.world_size > 1 or args.multiprocessing_distributed

#     ngpus_per_node = torch.cuda.device_count()
#     if args.multiprocessing_distributed:
#         # Since we have ngpus_per_node processes per node, the total world_size
#         # needs to be adjusted accordingly
#         args.world_size = ngpus_per_node * args.world_size
#         # Use torch.multiprocessing.spawn to launch distributed processes: the
#         # main_worker process function
#         mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
#     else:
#         # Simply call main_worker function
#         main_worker(args.gpu, ngpus_per_node, args)


# def main_worker(gpu, ngpus_per_node, args):
#     args.gpu = gpu

#     # suppress printing if not first GPU on each node
#     if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
#         def print_pass(*args):
#             pass
#         builtins.print = print_pass

#     if args.gpu is not None:
#         print("Use GPU: {} for training".format(args.gpu))
        
    
    
#     model = moco.builder.MoCo_ResNet(
#         partial(torchvision_models.__dict__[args.arch], zero_init_residual=True), 
#         args.moco_dim, args.moco_mlp_dim, args.moco_t)

#     # Assuming input_channels is set to args.geo_channels
#     input_channels = args.geo_channels
#     # Save original parameters of the first convolutional layer
#     out_channels = model.base_encoder.conv1.out_channels
#     kernel_size = model.base_encoder.conv1.kernel_size
#     stride = model.base_encoder.conv1.stride
#     padding = model.base_encoder.conv1.padding

#     # Create new convolutional layers with the required number of input channels
#     model.base_encoder.conv1 = nn.Conv2d(
#         input_channels, out_channels,
#         kernel_size=kernel_size,
#         stride=stride,
#         padding=padding,
#         bias=False
#     )

#     model.momentum_encoder.conv1 = nn.Conv2d(
#         input_channels, out_channels,
#         kernel_size=kernel_size,
#         stride=stride,
#         padding=padding,
#         bias=False
#     )
    
#     replace_batchnorm_with_identity(model) # заменяем все на identitiy
    

#     print("=> creating model '{}'".format(args.arch))
#     # Move to GPU if available
#     if torch.cuda.is_available():
#         model = model.cuda()

            
#     # infer learning rate before changing batch size
#     # args.lr = args.lr * args.batch_size / 256

#     if not torch.cuda.is_available():
#         print('using CPU, this will be slow')
#     elif args.distributed:
#         # apply SyncBN
#         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#         # For multiprocessing distributed, DistributedDataParallel constructor
#         # should always set the single device scope, otherwise,
#         # DistributedDataParallel will use all available devices.
#         if args.gpu is not None:
#             torch.cuda.set_device(args.gpu)
#             model.cuda(args.gpu)
#             # When using a single GPU per process and per
#             # DistributedDataParallel, we need to divide the batch size
#             # ourselves based on the total number of GPUs we have
#             args.batch_size = int(args.batch_size / args.world_size)
#             args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
#             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
#         else:
#             model.cuda()
#             # DistributedDataParallel will divide and allocate batch_size to all
#             # available GPUs if device_ids are not set
#             model = torch.nn.parallel.DistributedDataParallel(model)
#     elif args.gpu is not None:
#         torch.cuda.set_device(args.gpu)
#         model = model.cuda(args.gpu)
#         # comment out the following line for debugging
#         # raise NotImplementedError("Only DistributedDataParallel is supported.")
#     else:
#         # AllGather/rank implementation in this code only supports DistributedDataParallel.
#         raise NotImplementedError("Only DistributedDataParallel is supported.")
#     # print(model) # print model after SyncBatchNorm

#     if args.optimizer == 'lars':
#         optimizer = moco.optimizer.LARS(model.parameters(), args.lr,
#                                         weight_decay=args.weight_decay,
#                                         momentum=args.momentum)
#     elif args.optimizer == 'adamw':
#         optimizer = torch.optim.AdamW(model.parameters(), args.lr,
#                                 weight_decay=args.weight_decay)
        
#     scaler = torch.cuda.amp.GradScaler()
#     summary_writer = SummaryWriter() if ((args.rank == 0) or (args.rank == -1)) else None
    
#     # Запись аргументов в TensorBoard
#     if summary_writer is not None:
#         args_dict = vars(args)
#         args_text = '| Параметр | Значение |\n|----------|----------|\n'
#         args_text += '\n'.join([f'| **{k}** | `{v}` |' for k, v in args_dict.items()])
#         summary_writer.add_text('Аргументы обучения', args_text, 0)
#         summary_writer.flush()

#     # optionally resume from a checkpoint
#     if args.resume:
#         if os.path.isfile(args.resume):
#             print("=> loading checkpoint '{}'".format(args.resume))
#             if args.gpu is None:
#                 checkpoint = torch.load(args.resume)
#             else:
#                 # Map model to be loaded to specified single gpu.
#                 loc = 'cuda:{}'.format(args.gpu)
#                 checkpoint = torch.load(args.resume, map_location=loc)
#             args.start_epoch = checkpoint['epoch']
#             model.load_state_dict(checkpoint['state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             scaler.load_state_dict(checkpoint['scaler'])
#             print("=> loaded checkpoint '{}' (epoch {})"
#                   .format(args.resume, checkpoint['epoch']))
#         else:
#             print("=> no checkpoint found at '{}'".format(args.resume))

#     cudnn.benchmark = True


#     # normalize = transforms.Normalize(mean=[0.5], std=[1.0])

# # Пример использования
#     means = np.array([1.673302181686475, 33.37522164335293, 32.58433311325712, 
#             11.152242330669477, 0.025353081653846376, -0.00907171541589713, 
#             0.07366986763832623])
    
#     square_means = np.array([5.995956099912317, 1720.1733657260818, 1063.4138676153, 
#             149.60278359811343, 0.009805976106874816, 0.008356788111581723, 
#             0.035208209865639856])
#     stds = means**2 - square_means

#     augmentation2 = [
#         transforms.ToTensor(),
#         transforms.Normalize(mean=means, std=stds)
#     ]
#     augmentation1 = [
#         transforms.ToTensor(),
#         transforms.Normalize(mean=means, std=stds)
#     ]
#         # Инициализация датасета с параллельной загрузкой
#     train_dataset = moco.GLORYS12_dataset.Glorys12Dataset(
#         csv_file=args.csv_file,
#         num_io_workers=args.io_workers,
#         cache_size=args.cache_size,
#         transform1=transforms.Compose(augmentation1),
#         transform2=transforms.Compose(augmentation2)
#     )
#     print('dataset собрали')
#     # DataLoader с увеличенной производительностью
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.workers,
#         pin_memory=True,
#         drop_last=True
#     )
#     print('DataLoader собрали')

#     # В начале обучения запоминаем время запуска
#     start_time = datetime.now()
#     time_str = start_time.strftime("%Y%m%d_%H%M%S")
#     for epoch in range(args.start_epoch, args.epochs):
#         if args.distributed:
#             train_sampler.set_epoch(epoch)

#         # train for one epoch
#         print('start training')

#         train(train_loader, model, optimizer, scaler, summary_writer, epoch, args)
#         # if not args['multiprocessing_distributed'] or (args['multiprocessing_distributed'] and args['rank'] == 0):
#         state = {
#             'epoch': epoch + 1,
#             'arch': args.arch,
#             'state_dict': model.state_dict(),
#             'optimizer' : optimizer.state_dict(),
#             'scaler': scaler.state_dict(),
#         }
#         filename = f'/app/MoCo/MOCOv3-MNIST/checkpoints/{time_str}_checkpoint_{epoch:04d}.pth.tar'
#         torch.save(state, filename)


#     if ((args.rank == 0) or (args.rank == -1)):
#         summary_writer.close()

# def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
#     batch_time = AverageMeter('Time', ':6.3f')
#     data_time = AverageMeter('Data', ':6.3f')
#     learning_rates = AverageMeter('LR', ':.4e')
#     losses = AverageMeter('Loss', ':.4e')
#     progress = ProgressMeter(
#         len(train_loader),
#         [batch_time, data_time, learning_rates, losses],
#         prefix="Epoch: [{}]".format(epoch))

#     # switch to train mode
#     model.train()

#     end = time.time()
#     iters_per_epoch = len(train_loader)
#     moco_m = args.moco_m
#     # for i, (images, _) in enumerate(train_loader):
#     for i, images in enumerate(train_loader):
        
#         images = [img.float() for img in images] 
#         # measure data loading time
#         data_time.update(time.time() - end)

#         # adjust learning rate and momentum coefficient per iteration
#         # s
#         # lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
#         if args.moco_m_cos:
#             moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

#         if args.gpu is not None:
#             images[0] = images[0].cuda(args.gpu, non_blocking=True) 
#             images[1] = images[1].cuda(args.gpu, non_blocking=True)

#         # compute output
#         loss = model(images[0], images[1], moco_m)
#         losses.update(loss.item(), images[0].size(0))
#         if ((args.rank == 0) or (args.rank == -1)):
#             summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)

#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % args.print_freq == 0:
#             progress.display(i)


# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')

# def save_checkpoint(state, is_best, filename, start_time, checkpoint_dir='/app/MoCo/MOCOv3-MNIST/checkpoints'):
#     # Создаем директорию, если она не существует
#     os.makedirs(checkpoint_dir, exist_ok=True)
    
#     # Форматируем время запуска для использования в имени файла
#     time_str = start_time.strftime("%d%m%Y_%H%M%S")
    
#     # Определяем имя файла
#     filename = os.path.join(checkpoint_dir, f'checkpoint_{time_str}.pth.tar')
#     torch.save(state, filename)
    
#     # Копируем лучший чекпоинт
#     if is_best:
#         best_filename = os.path.join(checkpoint_dir, f'model_best_{time_str}.pth.tar')
# #         shutil.copyfile(filename, best_filename)
        
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self, name, fmt=':f'):
#         self.name = name
#         self.fmt = fmt
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

#     def __str__(self):
#         fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
#         return fmtstr.format(**self.__dict__)


# class ProgressMeter(object):
#     def __init__(self, num_batches, meters, prefix=""):
#         self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
#         self.meters = meters
#         self.prefix = prefix

#     def display(self, batch):
#         entries = [self.prefix + self.batch_fmtstr.format(batch)]
#         entries += [str(meter) for meter in self.meters]
#         print('\t'.join(entries))

#     def _get_batch_fmtstr(self, num_batches):
#         num_digits = len(str(num_batches // 1))
#         fmt = '{:' + str(num_digits) + 'd}'
#         return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# def adjust_learning_rate(optimizer, epoch, args):
#     """Decays the learning rate with half-cycle cosine after warmup"""
#     if epoch < args.warmup_epochs:
#         lr = args.lr * epoch / args.warmup_epochs 
#     else:
#         lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr


# def adjust_moco_momentum(epoch, args):
#     """Adjust moco momentum based on current epoch"""
#     m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
#     return m


# if __name__ == '__main__':
#     main()