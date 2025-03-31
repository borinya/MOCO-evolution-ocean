#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

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


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 20)')
    # Параметры данных
parser.add_argument('--io-workers', default=4, type=int, help='параллельная загрузка файлов')
parser.add_argument('--cache-size', default=512, type=int, help='размер кэша датасета')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    # Добавляем аргумент для CSV файла
parser.add_argument('-c', '--csv-file', default='/app/MoCo/MOCOv3-MNIST/momental files and code/test_file_pathes_dataset.csv',
                        type=str, metavar='FILE', help='path to the CSV file (default: /app/MoCo/MOCOv3-MNIST/momental files and code/test_file_pathes_dataset.csv)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=1.0, type=float,
                    help='softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='lars', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')

# geo_channels default 7))
parser.add_argument('--geo-channels', default=7, type=int,
                    help='nomber of input data channels (default: 7)')

def replace_batchnorm_with_identity(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.BatchNorm2d):
            setattr(model, name, torch.nn.Identity())
        else:
            replace_batchnorm_with_identity(module)

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        
    
    
    model = moco.builder.MoCo_ResNet(
        partial(torchvision_models.__dict__[args.arch], zero_init_residual=True), 
        args.moco_dim, args.moco_mlp_dim, args.moco_t)

    # Assuming input_channels is set to args.geo_channels
    input_channels = args.geo_channels
    # Save original parameters of the first convolutional layer
    out_channels = model.base_encoder.conv1.out_channels
    kernel_size = model.base_encoder.conv1.kernel_size
    stride = model.base_encoder.conv1.stride
    padding = model.base_encoder.conv1.padding

    # Create new convolutional layers with the required number of input channels
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
    
    replace_batchnorm_with_identity(model) # заменяем все на identitiy
    

    print("=> creating model '{}'".format(args.arch))
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

            
    # infer learning rate before changing batch size
    # args.lr = args.lr * args.batch_size / 256

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # print(model) # print model after SyncBatchNorm

    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
        
    scaler = torch.cuda.amp.GradScaler()
    summary_writer = SummaryWriter() if ((args.rank == 0) or (args.rank == -1)) else None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # normalize = transforms.Normalize(mean=[0.5], std=[1.0])

# Пример использования
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
        # Инициализация датасета с параллельной загрузкой
    train_dataset = moco.GLORYS12_dataset.Glorys12Dataset(
        csv_file=args.csv_file,
        num_io_workers=args.io_workers,
        cache_size=args.cache_size,
        transform1=transforms.Compose(augmentation1),
        transform2=transforms.Compose(augmentation2)
    )
    print('dataset собрали')
    # DataLoader с увеличенной производительностью
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    print('DataLoader собрали')
    

    
    # train_dataset = moco.GLORYS12_dataset.Glorys12Dataset(csv_file='/app/MoCo/MOCOv3-MNIST/momental files and code/test_file_pathes_dataset.csv',
    #                         transform1=transforms.Compose(augmentation1),
    #                         transform2=transforms.Compose(augmentation2))   #moco.loader.TwoCropsTransform(transforms.Compose(augmentation2))                                                                                            transforms.Compose(augmentation2)))
 
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle= True,
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    # В начале обучения запоминаем время запуска
    start_time = datetime.now()
    time_str = start_time.strftime("%Y%m%d_%H%M%S")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        print('start training')

        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args)
        # if not args['multiprocessing_distributed'] or (args['multiprocessing_distributed'] and args['rank'] == 0):
        state = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scaler': scaler.state_dict(),
        }
        filename = f'/app/MoCo/MOCOv3-MNIST/checkpoints/{time_str}_checkpoint_{epoch:04d}.pth.tar'
        torch.save(state, filename)


    if ((args.rank == 0) or (args.rank == -1)):
        summary_writer.close()

def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    # for i, (images, _) in enumerate(train_loader):
    for i, images in enumerate(train_loader):
        
        images = [img.float() for img in images] 
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        # s
        # lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True) 
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        loss = model(images[0], images[1], moco_m)
        losses.update(loss.item(), images[0].size(0))
        if ((args.rank == 0) or (args.rank == -1)):
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


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
#         shutil.copyfile(filename, best_filename)
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


if __name__ == '__main__':
    main()
