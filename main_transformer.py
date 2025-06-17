#!/usr/bin/env python

import argparse
import os
from datetime import datetime
from functools import partial
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torchvision.models import resnet50

# Локальные импорты
from moco.transformer_model import OceanTransformer
from moco.transformer_dataset import Glorys12SequenceDataset
def parse_args():
    parser = argparse.ArgumentParser(description='Ocean State Transformer Forecasting')
    
    # Пути
    parser.add_argument('--csv-file', type=str, default='/app/MoCo/MOCOv3-MNIST/momental files and code/cleaned_data.csv', 
                      help='Path to cleaned data CSV')
    parser.add_argument('--checkpoint', type=str, default='/app/MoCo/MOCOv3-MNIST/checkpoints/20250404_124558_checkpoint_0299.pth.tar',
                      help='Path to pretrained MoCo checkpoint')
    
    # Архитектура модели
    parser.add_argument('--transformer-layers', type=int, default=4,
                      help='Number of transformer layers')
    parser.add_argument('--transformer-heads', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--transformer-dim-ff', type=int, default=1024,
                      help='Feedforward dimension')
    parser.add_argument('--transformer-dropout', type=float, default=0.1,
                      help='Transformer dropout rate')
    parser.add_argument('--embedding-dim', type=int, default=512,
                      help='Dimension of transformer embeddings')
    
    # Параметры обучения
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
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                      help='Gradient accumulation steps')
    
    # Параметры данных
    parser.add_argument('--seq-len', type=int, default=60,
                      help='Input sequence length')
    parser.add_argument('--pred-horizon', type=int, default=30,
                      help='Prediction horizon')
    parser.add_argument('--predict-differences', action='store_true',
                      help='Predict differences instead of absolute values')
    parser.add_argument('--cache-size', type=int, default=512,
                      help='Dataset cache size')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of workers for DataLoader')
    
    # Управление экспериментом
    parser.add_argument('--amp', action='store_true',
                      help='Use mixed precision')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                      help='Gradient clipping norm')
    parser.add_argument('--log-dir', type=str, default='logs_transformer',
                      help='Base directory for logs')
    parser.add_argument('--save-interval', type=int, default=10,
                      help='Checkpoint saving interval')
    parser.add_argument('--val-split', type=float, default=0.1,
                      help='Validation set fraction')
    
    return parser.parse_args()

def setup_logging(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"transformer_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    # Сохраняем аргументы
    for arg in vars(args):
        writer.add_text(f"args/{arg}", str(getattr(args, arg)))
        
    return writer, log_dir

def load_moco_encoder(checkpoint_path, device, finetune=False):
    """
    Загрузка предобученного энкодера MoCo
    Ключевые изменения:
    1. Исправление загрузки состояния модели
    2. Автоматическое определение архитектуры из чекпоинта
    3. Гибкая настройка режима обучения
    """
    # Создаем базовый энкодер
    encoder = resnet50(pretrained=False)
    
    # Модификация первого слоя для 7 каналов
    encoder.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Удаление BatchNorm слоев
    def replace_batchnorm(module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.Identity())
            else:
                replace_batchnorm(child)
    
    replace_batchnorm(encoder)
    
    # Загрузка состояния
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Автоматическое определение префикса ключей
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if 'base_encoder' in k:
            # Нормализация ключей
            new_key = k.replace('module.base_encoder.', '').replace('base_encoder.', '')
            state_dict[new_key] = v
    
    # Загрузка весов
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    print(f"Loaded encoder: missing={len(missing)}, unexpected={len(unexpected)}")
    
    # Удаление последнего слоя (projection head)
    encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
    
    # Режим обучения
    encoder = encoder.to(device)
    if not finetune:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
    else:
        encoder.train()
    
    return encoder

class PositionalEncoding(nn.Module):
    """Позиционное кодирование для трансформера"""
    def __init__(self, d_model, max_len=180):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class OceanTransformer(nn.Module):
    """Улучшенная архитектура трансформера"""
    def __init__(self, input_dim=256, num_layers=4, nhead=8, 
                 dim_feedforward=1024, dropout=0.1, d_model=512):
        super().__init__()
        
        # Проекция признаков в пространство трансформера
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Позиционное кодирование
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Слои трансформера
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Регрессионная головка
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.input_proj(x)  # Проекция в d_model
        x = x.permute(1, 0, 2)  # [seq_len, batch, features]
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[-1]  # Берем последний временной шаг
        return self.regressor(x)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Настройка логирования
    writer, log_dir = setup_logging(args)
    print(f"Logs directory: {log_dir}")
    
    # Инициализация моделей
    encoder = load_moco_encoder(
        args.checkpoint, 
        device, 
        finetune=args.finetune_encoder
    )
    
    transformer = OceanTransformer(
        input_dim=2048,  # ResNet50 output features
        num_layers=args.transformer_layers,
        nhead=args.transformer_heads,
        dim_feedforward=args.transformer_dim_ff,
        dropout=args.transformer_dropout,
        d_model=args.embedding_dim
    ).to(device)
    
    # Оптимизатор
    params = [
        {'params': transformer.parameters(), 'lr': args.lr}
    ]
    if args.finetune_encoder:
        params.append({
            'params': encoder.parameters(), 
            'lr': args.encoder_lr
        })
    
    optimizer = optim.AdamW(params, weight_decay=args.weight_decay)
    
    # Нормализация данных (примерные значения)
    means = np.array([1.673, 33.375, 32.584, 11.152, 0.025, -0.009, 0.074])
    stds = np.array([0.125, 0.543, 0.621, 0.832, 0.015, 0.012, 0.042])
    
    augmentation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])
    
    # Подготовка данных
    full_dataset = Glorys12SequenceDataset(
        csv_file=args.csv_file,
        sequence_length=args.seq_len,
        prediction_horizon=args.pred_horizon,
        transform=augmentation,
        cache_size=args.cache_size
    )
    
    # Разделение на train/validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Обучение
    criterion = torch.nn.MSELoss()
    scaler = GradScaler(enabled=args.amp)
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        transformer.train()
        if args.finetune_encoder:
            encoder.train()
        
        train_loss = 0.0
        optimizer.zero_grad()
        
        for step, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Извлечение признаков
            with torch.set_grad_enabled(args.finetune_encoder):
                with autocast(enabled=args.amp):
                    # Переформатирование: [batch, seq, C, H, W] -> [batch*seq, C, H, W]
                    batch_size, seq_len = sequences.shape[:2]
                    features = sequences.view(-1, *sequences.shape[2:])
                    
                    # Пропуск через энкодер
                    features = encoder(features)
                    
                    # Возврат к последовательности: [batch, seq, features]
                    features = features.view(batch_size, seq_len, -1)
            
            # Прогнозирование трансформером
            with autocast(enabled=args.amp):
                predictions = transformer(features)
                loss = criterion(predictions.squeeze(), targets)
                loss = loss / args.grad_accum_steps
            
            # Обратное распространение с mixed precision
            scaler.scale(loss).backward()
            
            # Обновление весов с накоплением градиентов
            if (step + 1) % args.grad_accum_steps == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * args.grad_accum_steps
            global_step += 1
            
            if step % 10 == 0:
                writer.add_scalar('train/loss_step', loss.item(), global_step)
        
        # Validation phase
        val_loss = 0.0
        transformer.eval()
        encoder.eval()
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                # Извлечение признаков
                batch_size, seq_len = sequences.shape[:2]
                features = sequences.view(-1, *sequences.shape[2:])
                features = encoder(features)
                features = features.view(batch_size, seq_len, -1)
                
                # Прогнозирование
                predictions = transformer(features)
                loss = criterion(predictions.squeeze(), targets)
                val_loss += loss.item()
        
        # Логирование
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        writer.add_scalar('train/loss_epoch', avg_train_loss, epoch)
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'transformer': transformer.state_dict(),
                'encoder': encoder.state_dict() if args.finetune_encoder else None,
                'optimizer': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(log_dir, "best_model.pth"))
        
        # Периодическое сохранение
        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                checkpoint, 
                os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pth")
            )
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# import argparse
# import os
# import torch
# import torch.optim as optim 
# import torchvision.transforms as transforms

# import numpy as np
# from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
# from torch.cuda.amp import GradScaler, autocast

# from moco.transformer_model import OceanTransformer
# from moco.transformer_dataset import Glorys12SequenceDataset
# from torchvision.models import resnet50
# import torch.nn as nn
# from functools import partial



# def parse_args():
#     parser = argparse.ArgumentParser(description='Ocean State Transformer Forecasting')
    
#     # Paths
#     parser.add_argument('--csv-file', required=False, type=str, default='/app/MoCo/MOCOv3-MNIST/momental files and code/cleaned_data.csv', 
#                       help='Path to cleaned data CSV')
#     parser.add_argument('--checkpoint', required=False, type=str, default='/app/MoCo/MOCOv3-MNIST/checkpoints/20250404_124558_checkpoint_0299.pth.tar',
#                       help='Path to pretrained MoCo checkpoint')
    
#     # Model architecture
#     parser.add_argument('--transformer-layers', type=int, default=4,
#                       help='Number of transformer layers')
#     parser.add_argument('--transformer-heads', type=int, default=8,
#                       help='Number of attention heads')
#     parser.add_argument('--transformer-dim-ff', type=int, default=1024,
#                       help='Feedforward dimension')
#     parser.add_argument('--transformer-dropout', type=float, default=0.1,
#                       help='Transformer dropout rate')
    
#     # Training parameters
#     parser.add_argument('--epochs', type=int, default=100)
#     parser.add_argument('--batch-size', type=int, default=4)
#     parser.add_argument('--lr', type=float, default=3e-4,
#                       help='Base learning rate')
#     parser.add_argument('--encoder-lr', type=float, default=1e-5,
#                       help='Encoder learning rate if finetuning')
#     parser.add_argument('--weight-decay', type=float, default=1e-6)
#     parser.add_argument('--optimizer', choices=['adam', 'adamw'], default='adamw')
#     parser.add_argument('--finetune-encoder', action='store_true',
#                       help='Fine-tune the encoder')
    
#     # Data parameters
#     parser.add_argument('--seq-len', type=int, default=60,
#                       help='Input sequence length')
#     parser.add_argument('--pred-horizon', type=int, default=30,
#                       help='Prediction horizon')
#     parser.add_argument('--predict-differences', action='store_true',
#                       help='Predict differences instead of absolute values')
#     parser.add_argument('--transform', type=str, default=None,
#                       help='Type of transformation to apply')
#     parser.add_argument('--cache-size', type=int, default=512,
#                       help='Dataset cache size')
#     parser.add_argument('--num-io-workers', type=int, default=20,
#                       help='Number of IO workers for dataset preprocessing')
#     parser.add_argument('--prefetch-factor', type=int, default=2,
#                       help='Prefetch factor for data loading')
#     parser.add_argument('--random-seed', type=int, default=42,
#                       help='Random seed for reproducibility')
#     parser.add_argument('--num-workers', type=int, default=0,
#                       help='Number of workers for DataLoader')
    
#     # Experiment management
#     parser.add_argument('--amp', action='store_true',
#                       help='Use mixed precision')
#     parser.add_argument('--grad-clip', type=float, default=1.0,
#                       help='Gradient clipping norm')
#     parser.add_argument('--log-dir', type=str, default='/app/MoCo/logs_transformer', #/app/MoCo/MOCOv3-MNIST/runs_transformer
#                       help='Base directory for logs')
#     parser.add_argument('--save-interval', type=int, default=10,
#                       help='Checkpoint saving interval')
    
#     return parser.parse_args()

# def setup_logging(args):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_dir = os.path.join(args.log_dir, f"transformer_{timestamp}")
#     os.makedirs(log_dir, exist_ok=True)
    
#     writer = SummaryWriter(log_dir=log_dir)
#     # Save all arguments to tensorboard
#     for arg in vars(args):
#         writer.add_text(f"args/{arg}", str(getattr(args, arg)))
        
#     return writer, log_dir

# def load_moco_encoder(checkpoint_path):
#     # Создаем модифицированную архитектуру ResNet
#     encoder = resnet50(pretrained=False)
    
#     # Модификация первого сверточного слоя для 7 каналов
#     encoder.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
#     # Замена BatchNorm слоев на Identity
#     def replace_batchnorm(module):
#         for name, child in module.named_children():
#             if isinstance(child, nn.BatchNorm2d):
#                 setattr(module, name, nn.Identity())
#             else:
#                 replace_batchnorm(child)
    
#     replace_batchnorm(encoder)
    
#     # Загрузка весов из чекпоинта
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
#     # Извлечение весов только для базового энкодера
#     state_dict = {}
#     for k, v in checkpoint['state_dict'].items():
#         if k.startswith('base_encoder.'):
#             state_dict[k[len('base_encoder.'):]] = v
    
#     # Загрузка весов в модель
#     encoder.load_state_dict(state_dict, strict=True)
    
#     # Установка в eval режим
#     encoder.eval()
    
#     return encoder


# def main():
#     args = parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Setup logging and experiment tracking
#     writer, log_dir = setup_logging(args)
#     print(f"Experiment logs saved to: {log_dir}")
    
#     # Initialize models
#     encoder = load_moco_encoder(args.checkpoint, device, args.finetune_encoder)

#     transformer = OceanTransformer(
#         input_dim=256,
#         num_layers=args.transformer_layers,
#         nhead=args.transformer_heads,
#         dim_feedforward=args.transformer_dim_ff,
#         dropout=args.transformer_dropout
#     ).to(device)
    
#     # Optimizer setup
#     optim_params = [
#         {'params': transformer.parameters(), 'lr': args.lr}
#     ]
#     if args.finetune_encoder:
#         optim_params.append({'params': encoder.parameters(), 'lr': args.encoder_lr})
    
#     optimizer = optim.AdamW(optim_params, weight_decay=args.weight_decay) if args.optimizer == 'adamw' \
#         else optim.Adam(optim_params, weight_decay=args.weight_decay)
    
#     # нормализация данных
#     means = np.array([1.673302181686475, 33.37522164335293, 32.58433311325712, 
#             11.152242330669477, 0.025353081653846376, -0.00907171541589713, 
#             0.07366986763832623])
    
#     square_means = np.array([5.995956099912317, 1720.1733657260818, 1063.4138676153, 
#             149.60278359811343, 0.009805976106874816, 0.008356788111581723, 
#             0.035208209865639856])
#     stds = means**2 - square_means

#     augmentation =  transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=means, std=stds)
# ])

#     # Dataset and loader
#     dataset = Glorys12SequenceDataset(
#         csv_file=args.csv_file,
#         sequence_length=args.seq_len,
#         prediction_horizon=args.pred_horizon,
#         predict_differences=args.predict_differences,
#         transform=augmentation,
#         cache_size=args.cache_size,
#         num_io_workers=args.num_io_workers,
#         prefetch_factor=args.prefetch_factor,
#         random_seed=args.random_seed
#     )
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=True
#     )
    
#     # Training setup
#     criterion = torch.nn.MSELoss()
#     scaler = GradScaler(enabled=args.amp)
#     # scaler = torch.amp.GradScaler(device_type='cuda', enabled=args.amp)
#     global_step = 0
    
#     for epoch in range(args.epochs):
#         transformer.train()
#         if args.finetune_encoder:
#             encoder.train()
#         else:
#             encoder.eval()
            
#         epoch_loss = 0.0
        
#         for batch_idx, (sequences, targets) in enumerate(dataloader):
#             sequences = sequences.to(device, non_blocking=True)
#             targets = targets.to(device, non_blocking=True)
            
#             # Forward pass through encoder
#             with torch.set_grad_enabled(args.finetune_encoder):
#                 if args.amp:
#                     with autocast():
#                         batch_size, seq_len, H, W, C = sequences.shape
#                         features = encoder(sequences.view(-1, H, W, C).view(batch_size, seq_len, -1))
#                 else:
#                     batch_size, seq_len, H, W, C = sequences.shape
#                     features = encoder(sequences.view(-1, H, W, C)).view(batch_size, seq_len, -1)
            
#             # Transformer forward
#             optimizer.zero_grad()
            
#             if args.amp:
#                 with autocast():
#                     predictions = transformer(features)
#                     loss = criterion(predictions, targets)
#             else:
#                 predictions = transformer(features)
#                 loss = criterion(predictions, targets)
            
#             # Backward and optimize
#             scaler.scale(loss).backward()
            
#             if args.grad_clip > 0:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.grad_clip)
#                 if args.finetune_encoder:
#                     torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
            
#             scaler.step(optimizer)
#             scaler.update()
            
#             # Logging
#             epoch_loss += loss.item()
#             global_step += 1
#             writer.add_scalar('train/loss_step', loss.item(), global_step)
            
#         # Epoch logging
#         avg_loss = epoch_loss / len(dataloader)
#         writer.add_scalar('train/loss_epoch', avg_loss, epoch)
#         writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
#         print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}")
        
#         # Save checkpoint
#         if (epoch+1) % args.save_interval == 0:
#             checkpoint = {
#                 'epoch': epoch+1,
#                 'transformer': transformer.state_dict(),
#                 'encoder': encoder.state_dict() if args.finetune_encoder else None,
#                 'optimizer': optimizer.state_dict(),
#                 'args': vars(args),
#                 'loss': avg_loss
#             }
#             save_path = os.path.join(log_dir, f"c heckpoint_epoch_{epoch+1}.pth")
#             torch.save(checkpoint, save_path)
#             print(f"Saved checkpoint to {save_path}")
    
#     writer.close()

# if __name__ == "__main__":
#     main()
