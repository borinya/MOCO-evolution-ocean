# main_transformer_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from datetime import datetime
import argparse
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import re

class OceanDataset(Dataset):
    def __init__(self, directory, sequence_length=180, prediction_horizon=30, 
                 predict_differences=True, transform=None):
        """
        Args:
            directory (str): Путь к директории с файлами векторов
            sequence_length (int): Длина входной последовательности (в днях)
            prediction_horizon (int): Горизонт прогнозирования (в днях)
            predict_differences (bool): Прогнозировать разницу вместо абсолютных значений
            transform (callable): Трансформация для нормализации данных
        """
        self.directory = directory
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.predict_differences = predict_differences
        self.transform = transform
        self.data = self._load_and_sort_vectors()
        
    def _load_and_sort_vectors(self):
        """Загружает и сортирует все векторы из директории"""
        files = []
        for filename in os.listdir(self.directory):
            if filename.endswith('_features.npy'):
                # Извлекаем дату из имени файла
                match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
                if match:
                    date_str = match.group(1)
                    try:
                        date = datetime.strptime(date_str, '%Y-%m-%d')
                        files.append((date, filename))
                    except ValueError:
                        continue
        
        # Сортируем файлы по дате
        files.sort(key=lambda x: x[0])
        
        # Загружаем векторы в порядке возрастания даты
        vectors = []
        for _, filename in files:
            vector = np.load(os.path.join(self.directory, filename))
            vectors.append(vector)
        
        # Конвертируем в тензор [num_days, features]
        data_tensor = torch.tensor(np.array(vectors), dtype=torch.float32)
        
        # Применяем трансформацию (нормализацию) если задана
        if self.transform:
            data_tensor = self.transform(data_tensor)
            
        return data_tensor

    def __len__(self):
        """Общее количество последовательностей"""
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx):
        """
        Возвращает:
            sequence: тензор формы [sequence_length, features]
            target: тензор формы [features]
        """
        # Входная последовательность
        sequence = self.data[idx:idx + self.sequence_length]
        
        # Целевой вектор (через горизонт прогнозирования)
        target_idx = idx + self.sequence_length + self.prediction_horizon - 1
        target_vector = self.data[target_idx]
        
        if self.predict_differences:
            # Прогнозируем разницу между целевым и последним вектором последовательности
            target = target_vector - sequence[-1]
        else:
            # Прогнозируем абсолютное значение
            target = target_vector
            
        return sequence, target

def create_data_loaders(dataset, batch_size, train_ratio=0.8):
    """Создает DataLoader для обучения и валидации"""
    n = len(dataset)
    train_size = int(train_ratio * n)
    val_size = n - train_size
    
    # Разделяем данные без перемешивания (важно для временных рядов)
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, n))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,    # Перемешиваем батчи, но не временные последовательности
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, val_loader

def parse_args():
    parser = argparse.ArgumentParser(description="Transformer Model Training Configuration")
    
    # Общие параметры модели
    parser.add_argument('--model_type', type=str, default='Transformer', choices=['Transformer', 'Informer', 'Autoformer'], help='Тип модели')
    parser.add_argument('--input_size', type=int, default=256, help='Размер входного вектора фичей')
    parser.add_argument('--output_size', type=int, default=256, help='Размер выходного вектора прогноза')
    parser.add_argument('--sequence_length', type=int, default=180, help='Длина входной последовательности')  # Изменено на 180
    parser.add_argument('--prediction_horizon', type=int, default=30, help='Горизонт прогнозирования (в днях)')  # Изменено на 30
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
    parser.add_argument('--data_dir', type=str, default='/app/MoCo/MOCOv3-MNIST/features_glorys12_moco256/2025-07-07_moco256_20250216_141630_checkpoint_0202', help='Путь к данным')  # Обновлено
    parser.add_argument('--log_dir', type=str, default='/app/transformer/logs', help='Директория для логов')
    parser.add_argument('--model_dir', type=str, default='/app/transformer/models', help='Директория для моделей')
    parser.add_argument('--mean_std_file', type=str, default='/app/LSTM_salmon/mean_std.npy', help='Файл с нормализацией')
    
    
    
    
    parser.add_argument('--early_stop_patience', type=int, default=30, help='Ранняя остановка после N эпох без улучшений')

    
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
    mean = torch.tensor(mean_std['mean'], dtype=torch.float32)
    std = torch.tensor(mean_std['std'], dtype=torch.float32)
    
    # Создание трансформации для нормализации
    transform = transforms.Compose([
        transforms.Lambda(lambda x: (x - mean) / std)
    ])
    
    # Создание датасета
    dataset = OceanDataset(
        directory=args.data_dir,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        predict_differences=args.predict_differences,
        transform=transform
    )
    
    # Создание DataLoader
    train_loader, val_loader = create_data_loaders(
        dataset, 
        args.batch_size
    )
    
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
    
        # Обучение с ранней остановкой
    best_loss = float('inf')
    best_epoch = -1
    early_stop_counter = 0
    
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss = validate(model, val_loader, criterion, device, epoch, writer)
        
        # Обновление LR
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Проверка на улучшение
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'args': args_dict
            }, os.path.join(model_dir, 'best_model.pth'))
        else:
            early_stop_counter += 1
        
        print(f'Epoch {epoch+1}/{args.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_loss:.4f}')
        
        # Ранняя остановка
        if early_stop_counter >= args.early_stop_patience:
            print(f'Ранняя остановка на эпохе {epoch+1} (без улучшений {args.early_stop_patience} эпох)')
            break
    
    writer.close()
    return best_loss  # Возвращаем лучший лосс

if __name__ == "__main__":
    main()


