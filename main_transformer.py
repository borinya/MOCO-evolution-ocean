#!/usr/bin/env python

import argparse
import os
from datetime import datetime
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
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
    parser.add_argument('--num-workers', type=int, default=0,  # Уменьшено для избежания ошибок SHM
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
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU index to use (default: 0)')
    parser.add_argument('--predict-mean', action='store_true',
                      help='Predict mean value instead of full spatial field')
    
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
    Загрузка предобученного энкодера MoCo с исправлением размерностей
    """
    # Создаем базовый энкодер
    encoder = resnet50(weights=None)
    
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
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
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class OceanTransformer(nn.Module):
    """Улучшенная архитектура трансформера"""
    def __init__(self, input_dim=256, num_layers=4, nhead=8, 
                 dim_feedforward=1024, dropout=0.1, d_model=512, 
                 output_dim=7, spatial_output=False):
        super().__init__()
        self.spatial_output = spatial_output
        
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
        
        if spatial_output:
            # Декодер для пространственных данных
            self.decoder = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, 349 * 661 * output_dim)
            )
        else:
            # Регрессионная головка для среднего значения
            self.regressor = nn.Sequential(
                nn.Linear(d_model, d_model//2),
                nn.ReLU(),
                nn.Linear(d_model//2, output_dim)
            )

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.input_proj(x)  # Проекция в d_model
        x = x.permute(1, 0, 2)  # [seq_len, batch, features]
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[-1]  # Берем последний временной шаг
        
        if self.spatial_output:
            # Декодирование в пространственный тензор
            x = self.decoder(x)
            x = x.view(x.size(0), 7, 349, 661)  # [batch, channels, height, width]
            return x.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        else:
            return self.regressor(x)

def main():
    args = parse_args()
    
    # Выбор GPU
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Настройка логирования
    writer, log_dir = setup_logging(args)
    print(f"Logs directory: {log_dir}")
    
    # Инициализация моделей
    encoder = load_moco_encoder(
        args.checkpoint, 
        device, 
        finetune=args.finetune_encoder
    )
    
    # Определяем режим вывода
    spatial_output = not args.predict_mean
    
    transformer = OceanTransformer(
        input_dim=2048,  # ResNet50 output features
        num_layers=args.transformer_layers,
        nhead=args.transformer_heads,
        dim_feedforward=args.transformer_dim_ff,
        dropout=args.transformer_dropout,
        d_model=args.embedding_dim,
        output_dim=7,
        spatial_output=spatial_output
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
        predict_differences=args.predict_differences,
        transform=augmentation,
        cache_size=args.cache_size
    )
    
    # Разделение на train/validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # DataLoader с уменьшенным числом workers
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(args.num_workers, 2),  # Ограничиваем workers
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, 2)   # Ограничиваем workers
    )
    
    # Обучение
    criterion = torch.nn.MSELoss()
    scaler = GradScaler(device_type='cuda', enabled=args.amp)
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
            
            # Обработка целей в зависимости от режима
            if args.predict_mean:
                # Усредняем по пространственным измерениям
                targets = targets.mean(dim=(1, 2))  # [batch, 7]
            targets = targets.to(device, non_blocking=True)
            
            # Извлечение признаков
            with torch.set_grad_enabled(args.finetune_encoder):
                with autocast(device_type='cuda', enabled=args.amp):
                    # Переформатирование: [batch, seq, C, H, W] -> [batch*seq, C, H, W]
                    batch_size, seq_len, C, H, W = sequences.shape
                    features = sequences.view(-1, C, H, W)
                    
                    # Пропуск через энкодер
                    features = encoder(features)
                    
                    # Изменение view для совместимости с трансформером
                    features = features.view(batch_size, seq_len, -1)
            
            # Прогнозирование трансформером
            with autocast(device_type='cuda', enabled=args.amp):
                predictions = transformer(features)
                loss = criterion(predictions, targets)
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
                
                if args.predict_mean:
                    targets = targets.mean(dim=(1, 2))
                targets = targets.to(device)
                
                # Извлечение признаков
                batch_size, seq_len, C, H, W = sequences.shape
                features = sequences.view(-1, C, H, W)
                features = encoder(features)
                features = features.view(batch_size, seq_len, -1)
                
                # Прогнозирование
                predictions = transformer(features)
                loss = criterion(predictions, targets)
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
                'epoch': epoch+1,
                'transformer': transformer.state_dict(),
                'encoder': encoder.state_dict() if args.finetune_encoder else None,
                'optimizer': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'args': vars(args)
            }
            save_path = os.path.join(log_dir, "best_model.pth")
            torch.save(checkpoint, save_path)
            print(f"Saved best model with val loss: {avg_val_loss:.4f}")
        
        # Периодическое сохранение
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()