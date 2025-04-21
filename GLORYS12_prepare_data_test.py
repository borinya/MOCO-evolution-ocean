import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from moco.GLORYS12_dataset import Glorys12Dataset
import torchvision.transforms as transforms
from datetime import datetime

def analyze_data(csv_file, output_dir="data_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Загрузка CSV с датами
    df = pd.read_csv(csv_file, parse_dates=['Datetime'])
    dates = df['Datetime'].dt.strftime('%Y-%m-%d').tolist()
    file_paths = df['File Path'].tolist()

    # Конфигурация ускорения
    config = {
        'batch_size': 1,
        'num_workers': 0,
        'cache_size': 2,
        'num_io_workers': 1,
        'prefetch_factor': 4
    }

    # Инициализация датасетов
    def create_dataset(transform):
        return Glorys12Dataset(
            csv_file=csv_file,
            transform1=transform,
            transform2=None,
            num_io_workers=config['num_io_workers'],
            cache_size=config['cache_size'],
            prefetch_factor=config['prefetch_factor']
        )

    # Параметры нормализации
    means = np.array([1.6733, 33.3752, 32.5843, 11.1522, 0.02535, -0.00907, 0.07367])
    stds = np.sqrt(np.array([5.9959, 1720.17, 1063.41, 149.603, 0.009806, 0.008357, 0.03521]))

    # Создание датасетов
    raw_dataset = create_dataset(None)
    normalized_dataset = create_dataset(
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])
    )

    # Функция быстрого сбора статистики
    def fast_stats_analysis(dataset, name):
        loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=True,
            shuffle= False
        )

        # Исправленная инициализация хранилищ
        channel_stats = np.zeros((7, 3))  # [sum, sum_sq, count]
        file_stats = []
        
        with tqdm(total=len(loader), desc=f'{name} stats') as pbar:
            for batch_idx, batch in enumerate(loader):
                data = batch[0].numpy()
                # print('отладка12312')
                # Проверка и исправление размерности
                if data.ndim == 3:  # Если данные в формате [H, W, C]
                    # data = np.transpose(data, (0, 3, 1, 2))  # Конвертируем в [
                    batch_dates = dates[batch_idx*config['batch_size'] : (batch_idx+1)*config['batch_size']]

                    # Статистика по файлам
                    for i in range(data.shape[0]):
                        file_means = data[i].mean(axis=(1,2))
                        file_stats.append({
                            'date': batch_dates[i],
                            'means': file_means,
                            'stds': data[i].std(axis=(1,2)),
                            'file_path': file_paths[batch_idx*config['batch_size'] + i]
                        })

                    # Аккумулирующая статистика по каналам
                    channel_stats[:, 0] += data.sum(axis=(0,2,3))
                    channel_stats[:, 1] += (data**2).sum(axis=(0,2,3))
                    channel_stats[:, 2] += data[0].size // 7  # кол-во элементов на канал
                    pbar.update(1)

        # Расчет глобальных статистик
                    global_means = channel_stats[:, 0] / channel_stats[:, 2]
                    global_stds = np.sqrt(channel_stats[:, 1]/channel_stats[:, 2] - global_means**2)

        # Сохранение файловой статистики
        pd.DataFrame(file_stats).to_csv(os.path.join(output_dir, f'{name}_per_file_stats.csv'), index=False)

        return global_means, global_stds, file_stats

    # Анализ данных
    print("Анализ сырых данных...")
    raw_means, raw_stds, raw_files = fast_stats_analysis(raw_dataset, 'raw')
    
    print("\nАнализ нормализованных данных...")
    norm_means, norm_stds, norm_files = fast_stats_analysis(normalized_dataset, 'normalized')

    # Построение графиков с датами
    def plot_timeseries(stats, name):
        plt.figure(figsize=(25, 15))
        dates = [s['date'] for s in stats]
        
        for ch in range(7):
            plt.subplot(3, 3, ch+1)
            channel_means = [s['means'][ch] for s in stats]
            
            # Скользящее среднее
            window_size = 7
            rolling_mean = pd.Series(channel_means).rolling(window_size).mean()
            
            plt.plot(dates, channel_means, 'b.', alpha=0.3, label='Дневные значения')
            plt.plot(dates, rolling_mean, 'r-', label=f'{window_size}-дневное среднее')
            
            plt.title(f"Канал {ch} - {name} данные")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f'{name}_timeseries.png'), bbox_inches='tight')
        plt.close()

    # Генерация графиков
    print("\nГенерация графиков...")
    plot_timeseries(raw_files, 'raw')
    plot_timeseries(norm_files, 'normalized')

    # Сохранение финальной статистики
    stats_df = pd.DataFrame({
        'Channel': range(7),
        'Raw_Mean': raw_means,
        'Raw_Std': raw_stds,
        'Norm_Mean': norm_means,
        'Norm_Std': norm_stds
    })
    stats_df.to_csv(os.path.join(output_dir, 'global_stats.csv'), index=False)

    print(f"\nАнализ завершен. Результаты сохранены в: {output_dir}")

if __name__ == "__main__":
    csv_path = "/mnt/hippocamp/mborisov/MoCo/MOCOv3-MNIST/momental files and code/test_file_pathes_dataset.csv"
    analyze_data(csv_path)