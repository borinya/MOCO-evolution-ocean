import numpy as np
import os
import re
from datetime import datetime
from tqdm import tqdm

def compute_stats(data_dir, output_file):
    vectors = []
    files = []
    
    # Собираем все файлы
    for filename in os.listdir(data_dir):
        if filename.endswith('_features.npy'):
            match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            if match:
                date_str = match.group(1)
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    files.append((date, os.path.join(data_dir, filename)))
                except ValueError:
                    continue
    
    # Сортируем по дате
    files.sort(key=lambda x: x[0])
    
    # Загружаем все векторы
    for _, filepath in tqdm(files, desc="Loading files"):
        vec = np.load(filepath)
        vectors.append(vec)
    
    # Конвертируем в numpy array
    data = np.array(vectors)
    
    # Вычисляем статистики
    stats = {
        'mean': np.mean(data, axis=0).astype(np.float32),
        'std': np.std(data, axis=0).astype(np.float32),
        'median': np.median(data, axis=0).astype(np.float32),
        'min': np.min(data, axis=0).astype(np.float32),
        'max': np.max(data, axis=0).astype(np.float32),
        'shape': data.shape
    }
    
    # Сохраняем
    np.save(output_file, stats)
    print(f"Статистики сохранены в {output_file}")
    print(f"Данные: {data.shape[0]} векторов по {data.shape[1]} фичей")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Директория с векторами')
    parser.add_argument('--output_file', type=str, default='data_stats.npy',
                       help='Файл для сохранения статистик')
    args = parser.parse_args()
    
    compute_stats(args.data_dir, args.output_file)