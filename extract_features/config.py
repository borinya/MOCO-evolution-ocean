import os
import numpy as np
from datetime import datetime

# Нормализация для GLORYS12
MEANS = np.array([1.673302181686475, 33.37522164335293, 32.58433311325712,
                  11.152242330669477, 0.025353081653846376, -0.00907171541589713,
                  0.07366986763832623])
SQUARE_MEANS = np.array([5.995956099912317, 1720.1733657260818, 1063.4138676153,
                         149.60278359811343, 0.009805976106874816, 0.008356788111581723,
                         0.035208209865639856])
STDS = MEANS**2 - SQUARE_MEANS

BATCH_SIZE = 50
MOCO_DIM = 2

VARIABLES = ['mlotst', 'thetao', 'bottomT', 'uo', 'vo', 'so', 'zos']
DATA_SHAPE = (7, 349, 661)

CHECKPOINT_PATH = '/app/MoCo/MOCOv3-MNIST/checkpoints/20250712_103622_checkpoint_0299.pth.tar'
CSV_FILE = '/app/MoCo/MOCOv3-MNIST/momental files and code/test_file_pathes_dataset.csv'
FEATURES_DIR = '/app/MoCo/MOCOv3-MNIST/features_glorys12_moco256'
 # /app/MoCo/MOCOv3-MNIST/checkpoints/20250711_160100_checkpoint_0299.pth.tar это 7- дней и 45 для + и в 256 вектор вроде # 
 # '/app/MoCo/MOCOv3-MNIST/checkpoints/20250216_141630_checkpoint_0202.pth.tar'  основной лучший ровде бы
 
 # /app/MoCo/MOCOv3-MNIST/checkpoints/20250712_103622_checkpoint_0299.pth.tar 2d на выходе
def make_experiment_tag():
    today = datetime.now().strftime('%Y-%m-%d')
    ckpt_tail = os.path.basename(CHECKPOINT_PATH).replace('.pth.tar','')
    return f"{today}_moco256_{ckpt_tail}"

EXPERIMENT_TAG = make_experiment_tag()