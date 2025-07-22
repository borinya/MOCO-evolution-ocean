# main_transformer_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from datetime import datetime
import argparse
import importlib

def parse_args():
    parser = argparse.ArgumentParser(description="Transformer Model Training Configuration")
    
    # Общие параметры модели
    parser.add_argument('--model_type', type=str, default='Transformer', choices=['Transformer', 'Informer', 'Autoformer'], help='Тип модели')
    parser.add_argument('--input_size', type=int, default=256, help='Размер входного вектора фичей')
    parser.add_argument('--output_size', type=int, default=256, help='Размер выходного вектора прогноза')
    parser.add_argument('--sequence_length', type=int, default=30, help='Длина входной последовательности')
    parser.add_argument('--prediction_horizon', type=int, default=1, help='Горизонт прогнозирования (в днях)')
    parser.add_argument('--predict_differences', type=bool, default=True, help='Прогнозировать разницы вместо абсолютных значений')
    
    # Параметры Transformer
    parser.add_argument('--d_model', type=int, default=512, help='Размерность эмбеддингов')
    parser.add_argument('--nhead', type=int, default=8, help='Количество голов внимания')
    parser.add_argument('--num_layers', type=int, default=4, help='Количество слоев энкодера/декодера')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Размерность FFN слоя')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    
    # Параметры обучения
    parser.add_argument('--num_epochs', type=int, default=300, help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=128, help='Размер батча')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Скорость обучения')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Вес decay')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['step', 'cosine', 'plateau'], help='Тип расписания LR')
    
    # Пути и логирование
    parser.add_argument('--data_dir', type=str, default='/app/LSTM_salmon/ocean_vectors', help='Путь к данным')
    parser.add_argument('--log_dir', type=str, default='/app/transformer/logs', help='Директория для логов')
    parser.add_argument('--model_dir', type=str, default='/app/transformer/models', help='Директория для моделей')
    parser.add_argument('--mean_std_file', type=str, default='/app/LSTM_salmon/mean_std.npy', help='Файл с нормализацией')
    
    return parser.parse_args()

class OceanTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = nn.Linear(args.input_size, args.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(args.d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, args.output_size)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x[:, -1, :]  # Берем последний временной шаг
        return self.decoder(x)

def train(model, data_loader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    return avg_loss

def validate(model, data_loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    writer.add_scalar('Loss/val', avg_loss, epoch)
    return avg_loss

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создание директорий
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, timestamp)
    model_dir = os.path.join(args.model_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Инициализация TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # Запись параметров
    args_dict = vars(args)
    writer.add_text('Parameters', str(args_dict))
    
    # Загрузка нормализации
    mean_std = np.load(args.mean_std_file, allow_pickle=True).item()
    mean = mean_std['mean']
    std = mean_std['std']
    
    # Загрузка данных
    transform = transforms.Lambda(lambda x: (torch.tensor(x, dtype=torch.float32) - mean) / std)
    dataset = OceanDataset(
        args.data_dir,
        prediction_horizon=args.prediction_horizon,
        sequence_length=args.sequence_length,
        predict_differences=args.predict_differences,
        transform=transform
    )
    train_loader, val_loader = create_data_loaders(dataset, args.batch_size)
    
    # Инициализация модели
    model = OceanTransformer(args).to(device)
    
    # Функция потерь и оптимизатор
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Расписание LR
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:  # plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    
    # Обучение
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss = validate(model, val_loader, criterion, device, epoch, writer)
        
        # Обновление LR
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Сохранение лучшей модели
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'args': args_dict
            }, os.path.join(model_dir, 'best_model.pth'))
        
        print(f'Epoch {epoch+1}/{args.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    
    writer.close()

if __name__ == "__main__":
    main()