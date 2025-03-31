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
# Пример использования

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
train_loader = torch.utils.data.DataLoader(
train_dataset, batch_size=batch_size, shuffle=None,
num_workers=20, pin_memory=True, drop_last=True)

# print('startamnuli')
# for i, images in enumerate(train_loader):
#     print (len(images), len(images[0]))
#     # measure data loading time
#     if i >6:
#         break

import matplotlib.pyplot as plt
import torch

# Предположим, train_loader — это ваш DataLoader.
# Выводим информацию о размерах изображений
print('Запустили')
image_sizes = []

# Запуск цикла по батчам
image_sizes = []

for i, images in enumerate(train_loader):
    # Предположим, что images[0] - это тензор изображений
    print(f"Итерация {i}:")
    
    # Получаем тип и размер изображений
    print(f"Тип: {type(images)}, Размер: {len(images)}")
    
    # Проверяем, является ли первый элемент тензором
    if isinstance(images[0], torch.Tensor):
        print(f"Тип изображений: {type(images[0])}, Размер: {images[0].size()}")
        
        # Сохраняем размеры изображений для графика
        image_sizes.append(images[0].size())
    
    # Измеряем время загрузки данных
    if i > batch_size:
        break

# Извлекаем ширину и высоту изображений
widths = [size[2] for size in image_sizes]  # предполагается, что размер тензора (N, C, H, W)
heights = [size[1] for size in image_sizes]

# Отрисовка графика
plt.figure(figsize=(10, 6))
plt.scatter(widths, heights, color='blue')
plt.title('Размеры изображений в батче')
plt.xlabel('Ширина')
plt.ylabel('Высота')
plt.grid()
plt.show()
