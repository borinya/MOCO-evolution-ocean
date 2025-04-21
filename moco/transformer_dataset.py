import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from netCDF4 import Dataset as netCDF4_Dataset

class Glorys12SequenceDataset(Dataset):
    def __init__(
        self,
        csv_file,
        sequence_length=60,
        prediction_horizon=30,
        predict_differences=False,
        transform=None,
        cache_size=512,
        num_io_workers=20,
        prefetch_factor=2,
        random_seed=42
    ):
        # Инициализация параметров
        self.data_frame = pd.read_csv(csv_file)
        self.file_paths = self.data_frame['File Path'].tolist()
        self.dates = pd.to_datetime(self.data_frame['Date'])
        
        # Временные параметры
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.predict_differences = predict_differences
        self.transform = transform

        # Параметры кэширования и многопоточности
        self.cache = {}
        self.cache_size = cache_size
        self.num_io_workers = num_io_workers
        self.prefetch_factor = prefetch_factor
        self.executor = ThreadPoolExecutor(max_workers=num_io_workers)
        self.pending = {}

        # Конфигурация переменных netCDF
        self.variables = {
            'mlotst': (0,),
            'thetao': (0, 0),
            'bottomT': (0,),
            'uo': (0, 0),
            'vo': (0, 0),
            'so': (0, 0),
            'zos': (0,)
        }

    def __len__(self):
        total_items = len(self.file_paths)
        return max(0, total_items - self.sequence_length - self.prediction_horizon)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()
            
        # Загрузка последовательности
        sequence = []
        for i in range(self.sequence_length):
            item_idx = idx + i
            if item_idx >= len(self.file_paths):
                raise IndexError(f"Index {item_idx} out of range")
                
            if item_idx in self.cache:
                data = self.cache[item_idx]
            else:
                data = self._load_single_file(item_idx)
                self._update_cache(item_idx, data)
            # data = torch.tensor(data, dtype=torch.float32)
            if self.transform:
                data = self.transform(data)
            sequence.append(data)

        # Загрузка целевого поля
        target_idx = idx + self.sequence_length + self.prediction_horizon - 1
        if target_idx >= len(self.file_paths):
            raise IndexError(f"Target index {target_idx} out of range")
            
        if target_idx in self.cache:
            target = self.cache[target_idx]
        else:
            target = self._load_single_file(target_idx)
            self._update_cache(target_idx, target)

        target = torch.tensor(target, dtype=torch.float32)
        
        if self.predict_differences:
            target -= sequence[-1]

        return torch.stack(sequence), target

    def _load_single_file(self, idx):
        file_path = self.file_paths[idx]
        data_array = np.zeros((len(self.variables), 349, 661), dtype=np.float32)
        
        try:
            with netCDF4_Dataset(file_path, 'r') as ds:
                for i, (var_name, index) in enumerate(self.variables.items()):
                    if var_name in ds.variables:
                        var = ds[var_name]
                        variable_data = np.array(var[index], dtype=np.float32)
                        variable_data = np.squeeze(variable_data)
                        variable_data = np.where(
                            variable_data == var._FillValue, 
                            np.nan, 
                            variable_data
                        )
                        data_array[i] = variable_data

                result = data_array.transpose((1, 2, 0))
                return np.nan_to_num(result, nan=0.0)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise

    def _update_cache(self, idx, data):
        if len(self.cache) >= self.cache_size:
            del self.cache[next(iter(self.cache))]
        self.cache[idx] = data

    def __del__(self):
        self.executor.shutdown()

# # Пример использования
# dataset = Glorys12SequenceDataset(
#     csv_file='/app/MoCo/MOCOv3-MNIST/momental files and code/cleaned_data.csv',
#     sequence_length=10,
#     prediction_horizon=33
# )

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
# print('DataLoader собран успешно')

# for i, (sequences, targets) in enumerate(dataloader):
#     print(f"Тип данных: {type(sequences)}")
#     print(f"Размерность батча: {sequences.shape}")
#     break
# import os
# import numpy as np
# import torch
# import pandas as pd
# from datetime import datetime
# from torch.utils.data import Dataset
# from concurrent.futures import ThreadPoolExecutor
# from netCDF4 import Dataset as netCDF4_Dataset
# import h5py





# class Glorys12SequenceDataset(Dataset):
#     def __init__(
#         self,
#         csv_file,
#         # start_month_day='01-01',
#         # end_month_day='12-31',
#         sequence_length=60,
#         prediction_horizon=30,
#         predict_differences=False, ## это после рассчитывать 
#         transform=None,
#         cache_size=512,
#         num_io_workers=20,
#         prefetch_factor=2,
#         random_seed=42
#     ):
#         # Инициализация параметров
#         self.data_frame = pd.read_csv(csv_file)
#         self.file_paths = self.data_frame['File Path'].tolist()
#         self.dates = pd.to_datetime(self.data_frame['Date'])
        
#         # Временные параметры
#         self.start_month, self.start_day = map(int, start_month_day.split('-'))
#         self.end_month, self.end_month_day = map(int, end_month_day.split('-'))
#         self.sequence_length = sequence_length
#         self.prediction_horizon = prediction_horizon
#         self.predict_differences = predict_differences
#         self.transform = transform

#         # Фильтрация данных по сезону
#         # self.filtered_indices = self._filter_by_season()
#         self.filtered_indices = self._filter_by_season()

#         # Параметры кэширования и многопоточности
#         self.cache = {}
#         self.cache_size = cache_size
#         self.num_io_workers = num_io_workers
#         self.prefetch_factor = prefetch_factor
#         self.executor = ThreadPoolExecutor(max_workers=num_io_workers)
#         self.pending = {}

#         # Конфигурация переменных netCDF
#         self.variables = {
#             'mlotst': (0,),
#             'thetao': (0, 0),
#             'bottomT': (0,),
#             'uo': (0, 0),
#             'vo': (0, 0),
#             'so': (0, 0),
#             'zos': (0,)
#         }

#     def _filter_by_season(self):
#         indices = []
#         for idx, date in enumerate(self.dates):
#             if (date.month > self.start_month or 
#                (date.month == self.start_month and date.day >= self.start_day)) and \
#                (date.month < self.end_month or 
#                (date.month == self.end_month and date.day <= self.end_month_day)):
#                 indices.append(idx)
#         return indices

#     def __len__(self):
#         return len(self.filtered_indices) - self.sequence_length - self.prediction_horizon

#     def __getitem__(self, idx):
#         # Получение реальных индексов из отфильтрованных
#         real_idx = self.filtered_indices[idx]
        
#         # Загрузка последовательности
#         sequence = []
#         for i in range(self.sequence_length):
#             item_idx = real_idx + i
#             if item_idx in self.cache:
#                 data = self.cache[item_idx]
#             else:
#                 data = self._load_single_file(item_idx)
#                 self._update_cache(item_idx, data)
            
#             if self.transform:
#                 data = self.transform(data)
#             sequence.append(torch.tensor(data, dtype=torch.float32))

#         # Загрузка целевого поля
#         target_idx = real_idx + self.sequence_length + self.prediction_horizon - 1
#         if target_idx in self.cache:
#             target = self.cache[target_idx]
#         else:
#             target = self._load_single_file(target_idx)
#             self._update_cache(target_idx, target)

#         target = torch.tensor(target, dtype=torch.float32)
        
#         if self.predict_differences:
#             target -= sequence[-1]

#         return torch.stack(sequence),  target

#     def _load_single_file(self, idx):
#         # Ваш существующий код загрузки из Glorys12Dataset
#         file_path = self.file_paths[idx]
#         data_array = np.zeros((len(self.variables), 349, 661), dtype=np.float32)
        
#         try:
#             with netCDF4_Dataset(file_path, 'r') as ds:
#                 for i, (var_name, index) in enumerate(self.variables.items()):
#                     if var_name in ds.variables:
#                         var = ds[var_name]
#                         variable_data = np.array(var[index], dtype=np.float32)
#                         variable_data = np.squeeze(variable_data)
#                         variable_data = np.where(
#                             variable_data == var._FillValue, 
#                             np.nan, 
#                             variable_data
#                         )
#                         data_array[i] = variable_data

#                 result = data_array.transpose((1, 2, 0))
#                 return np.nan_to_num(result, nan=0.0)
#         except Exception as e:
#             print(f"Error loading {file_path}: {str(e)}")
#             raise

#     def _update_cache(self, idx, data):
#         if len(self.cache) >= self.cache_size:
#             del self.cache[next(iter(self.cache))]
#         self.cache[idx] = data

#     def _async_prefetch(self, idx):
#         # Реализация асинхронной предзагрузки
#         pass

#     def __del__(self):
#         self.executor.shutdown()
            
# dataset = Glorys12SequenceDataset(
#     csv_file='/app/MoCo/MOCOv3-MNIST/momental files and code/cleaned_data.csv',
#     start_month_day='12-01',
#     end_month_day='02-28',
#     sequence_length=10,
#     prediction_horizon=33
# )      
        
# dataloader =  torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
# print('DataLoader собрали')
# for i, images in enumerate(dataloader):
#     print(type(images), images.shape())
#     break




    # images = [img.float() for img in images] 
# import os
# import random
# import threading
# from concurrent.futures import ThreadPoolExecutor
# import numpy as np
# import pandas as pd
# import torch
# from netCDF4 import Dataset as netCDF4_Dataset
# from torch.utils.data import Dataset
# import h5py

# class Glorys12Dataset(Dataset):
#     def __init__(
#         self, 
#         csv_file, 
#         transform1=None, 
#         transform2=None, 
#         random_seed=42,
#         delta_days=15,
#         cache_size=512,  #
#         num_io_workers=20,  # Количество параллельных IO workers
#         prefetch_factor=2  # Предзагрузка следующих элементов
#     ):
#         # Валидация входных параметров
#         if not os.path.exists(csv_file):
#             raise FileNotFoundError(f"CSV file {csv_file} not found")
            
#         self.data_frame = pd.read_csv(csv_file)
#         if len(self.data_frame) == 0:
#             raise ValueError("CSV file is empty")

#         # Инициализация параметров
#         self.file_paths = self.data_frame['File Path'].tolist()
#         # for fp in self.file_paths:
#         #     if not os.path.exists(fp):
#         #         raise FileNotFoundError(f"File {fp} not found")
#         self.transform1 = transform1
#         self.transform2 = transform2
#         self.delta_days = delta_days
#         self.cache_size = cache_size
#         self.num_io_workers = num_io_workers
#         self.read_lock = threading.Lock()

#         # Конфигурация переменных netCDF
#         self.variables = {
#             'mlotst': (0,),
#             'thetao': (0, 0),
#             'bottomT': (0,),
#             'uo': (0, 0),
#             'vo': (0, 0),
#             'so': (0, 0),
#             'zos': (0,)
#         }

#         # Инициализация системы кэширования
#         self.cache = {}
#         self.cache_lock = threading.Lock()
#         self.pending_futures = {}
        
#         # Пул потоков для параллельной загрузки
#         self.io_executor = ThreadPoolExecutor(max_workers=num_io_workers)
        
#         # Система предзагрузки
#         self.prefetch_factor = prefetch_factor
        
#         # Инициализация случайных состояний
#         if random_seed is not None:
#             random.seed(random_seed)
#             np.random.seed(random_seed)
#             torch.manual_seed(random_seed)

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
        
#         close_idx = self._random_close_idx(idx)
#         # Предзагрузка следующих элементов
#         self._prefetch_adjacent(idx)
#         self._prefetch_adjacent(close_idx)
#         # Получение данных с кэшированием
#         data_array1 = self._get_cached_data(idx)
#         data_array2 = self._get_cached_data(close_idx)
#         # Генерация аугментированных данных
#         if self.transform1 and self.transform2:

#             # data_array1 = self._load_single_file(idx)   # загружать напрямую
#             # data_array2 = self._load_single_file(close_idx)  

#             data_array1 = self.transform1(data_array1)
#             data_array2 = self.transform2(data_array2)
#         return data_array1, data_array2

#     def _prefetch_adjacent(self, idx):
#         """Асинхронная предзагрузка соседних индексов"""
#         for offset in range(1, self.prefetch_factor + 1):
#             next_idx = idx + offset
#             if next_idx < len(self):
#                 self._async_load(next_idx)

#     def _async_load(self, idx):
#         """Асинхронная загрузка данных в кэш"""
#         with self.cache_lock:
#             if idx in self.cache or idx in self.pending_futures:
#                 return

#         future = self.io_executor.submit(self._load_single_file, idx)
#         self.pending_futures[idx] = future

#     def _get_cached_data(self, idx):
#         """Получение данных с использованием кэша"""
#         # Проверка кэша
#         with self.cache_lock:
#             if idx in self.cache:
#                 # print(f"Cache hit for index {idx}")  # Логирование
#                 return self.cache[idx]

#         # Проверка асинхронной загрузки
#         if idx in self.pending_futures:
#             data = self.pending_futures[idx].result()
#             with self.cache_lock:
#                 self._update_cache(idx, data)
#                 del self.pending_futures[idx]
#             return data

#         # Синхронная загрузка при промахе кэша
#         return self._load_single_file(idx)

#     def _load_single_file(self, idx):
#         """Основной метод загрузки файла"""
#         # print(f"Loading from disk: {idx}")  # Логирование
#         file_path = self.file_paths[idx]
#         data_array = np.zeros((len(self.variables), 349, 661), dtype=np.float32)
        
#         try:
#             with self.read_lock:  # Блокировка доступа
#                 with netCDF4_Dataset(file_path, 'r') as ds:
#                     for i, (var_name, index) in enumerate(self.variables.items()):
#                         if var_name in ds.variables:
#                             var = ds[var_name]
#                             variable_data = np.array(var[index], dtype=np.float32)
#                             variable_data = np.squeeze(variable_data)
#                             variable_data = np.where(
#                                 variable_data == var._FillValue, 
#                                 np.nan, 
#                                 variable_data
#                             )
#                             data_array[i] = variable_data

#                     result = data_array.transpose((1, 2, 0))
#                     result = np.nan_to_num(result, nan=0.0)

#         except Exception as e:
#             print(f"Error loading {file_path}: {str(e)}")
#             raise

#         with self.cache_lock:
#             self._update_cache(idx, result)

#         return result

#     def _update_cache(self, idx, data):
#         """Обновление кэша с учетом LRU политики"""
#         if len(self.cache) >= self.cache_size:
#             # Удаляем самый старый элемент
#             del self.cache[next(iter(self.cache))]
#         self.cache[idx] = data

#     def _random_close_idx(self, idx):
#         """Генерация случайного близкого индекса"""
#         start = max(0, idx - self.delta_days)
#         end = min(len(self), idx + self.delta_days + 1)
#         return random.choice([i for i in range(start, end) if i != idx])

#     def __del__(self):
#         """Корректное завершение при удалении объекта"""
#         self.io_executor.shutdown(wait=True)




# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from moco.GLORYS12_dataset import Glorys12Dataset
# import torchvision.transforms as transforms

# class OnlineOceanDataset(Dataset):
#     def __init__(
#         self,
#         csv_file,
#         encoder,
#         seq_length=30,
#         prediction_offset=30,
#         transform=None,
#         cache_size=512
#     ):
#         self.base_dataset = Glorys12Dataset(
#             csv_file=csv_file,
#             transform1=transform,
#             transform2=transform,
#             cache_size=cache_size
#         )
#         self.encoder = encoder
#         self.seq_length = seq_length
#         self.prediction_offset = prediction_offset
        
#         # Кэш для скрытых представлений
#         self.feature_cache = {}
        
#     def __len__(self):
#         return len(self.base_dataset) - self.seq_length - self.prediction_offset

#     def __getitem__(self, idx):
#         # Получение последовательности и цели
#         input_seq = []
#         for i in range(self.seq_length):
#             item_idx = idx + i
#             if item_idx in self.feature_cache:
#                 input_seq.append(self.feature_cache[item_idx])
#             else:
#                 data, _ = self.base_dataset[item_idx]
#                 with torch.no_grad():
#                     features = self.encoder(data.unsqueeze(0)).squeeze()
#                 self.feature_cache[item_idx] = features
#                 input_seq.append(features)
        
#         # Получение цели (30 дней вперед)
#         target_idx = idx + self.seq_length + self.prediction_offset - 1
#         if target_idx in self.feature_cache:
#             target = self.feature_cache[target_idx]
#         else:
#             data, _ = self.base_dataset[target_idx]
#             with torch.no_grad():
#                 target = self.encoder(data.unsqueeze(0)).squeeze()
#             self.feature_cache[target_idx] = target
        
#         return torch.stack(input_seq), target