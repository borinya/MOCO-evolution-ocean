import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset as netCDF4_Dataset
from torch.utils.data import Dataset
import h5py

class Glorys12Dataset(Dataset):
    def __init__(
        self, 
        csv_file, 
        transform1=None, 
        transform2=None, 
        random_seed=42,
        delta_days=15,
        cache_size=512,  #
        num_io_workers=20,  # Количество параллельных IO workers
        prefetch_factor=2  # Предзагрузка следующих элементов
    ):
        # Валидация входных параметров
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file {csv_file} not found")
            
        self.data_frame = pd.read_csv(csv_file)
        if len(self.data_frame) == 0:
            raise ValueError("CSV file is empty")

        # Инициализация параметров
        self.file_paths = self.data_frame['File Path'].tolist()
        # for fp in self.file_paths:
        #     if not os.path.exists(fp):
        #         raise FileNotFoundError(f"File {fp} not found")
        self.transform1 = transform1
        self.transform2 = transform2
        self.delta_days = delta_days
        self.cache_size = cache_size
        self.num_io_workers = num_io_workers
        self.read_lock = threading.Lock()

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

        # Инициализация системы кэширования
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.pending_futures = {}
        
        # Пул потоков для параллельной загрузки
        self.io_executor = ThreadPoolExecutor(max_workers=num_io_workers)
        
        # Система предзагрузки
        self.prefetch_factor = prefetch_factor
        
        # Инициализация случайных состояний
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        
        close_idx = self._random_close_idx(idx)
        # Предзагрузка следующих элементов
        self._prefetch_adjacent(idx)
        self._prefetch_adjacent(close_idx)
        # Получение данных с кэшированием
        data_array = self._get_cached_data(idx)
        
        # Генерация аугментированных данных
        if self.transform1 and self.transform2:
            data_array1 = self._get_cached_data(idx)
            data_array2 = self._get_cached_data(close_idx)
            # data_array1 = self._load_single_file(idx)   # загружать напрямую
            # data_array2 = self._load_single_file(close_idx)  

            if self.transform1 and self.transform2:
                data_array1 = self.transform1(data_array1)
                data_array2 = self.transform2(data_array2)
        return data_array1, data_array2

    def _prefetch_adjacent(self, idx):
        """Асинхронная предзагрузка соседних индексов"""
        for offset in range(1, self.prefetch_factor + 1):
            next_idx = idx + offset
            if next_idx < len(self):
                self._async_load(next_idx)

    def _async_load(self, idx):
        """Асинхронная загрузка данных в кэш"""
        with self.cache_lock:
            if idx in self.cache or idx in self.pending_futures:
                return

        future = self.io_executor.submit(self._load_single_file, idx)
        self.pending_futures[idx] = future

    def _get_cached_data(self, idx):
        """Получение данных с использованием кэша"""
        # Проверка кэша
        with self.cache_lock:
            if idx in self.cache:
                # print(f"Cache hit for index {idx}")  # Логирование
                return self.cache[idx]

        # Проверка асинхронной загрузки
        if idx in self.pending_futures:
            data = self.pending_futures[idx].result()
            with self.cache_lock:
                self._update_cache(idx, data)
                del self.pending_futures[idx]
            return data

        # Синхронная загрузка при промахе кэша
        return self._load_single_file(idx)

    def _load_single_file(self, idx):
        """Основной метод загрузки файла"""
        # print(f"Loading from disk: {idx}")  # Логирование
        file_path = self.file_paths[idx]
        data_array = np.zeros((len(self.variables), 349, 661), dtype=np.float32)
        
        try:
            with self.read_lock:  # Блокировка доступа
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
                    result = np.nan_to_num(result, nan=0.0)

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise

        with self.cache_lock:
            self._update_cache(idx, result)

        return result

    def _update_cache(self, idx, data):
        """Обновление кэша с учетом LRU политики"""
        if len(self.cache) >= self.cache_size:
            # Удаляем самый старый элемент
            del self.cache[next(iter(self.cache))]
        self.cache[idx] = data

    def _random_close_idx(self, idx):
        """Генерация случайного близкого индекса"""
        start = max(0, idx - self.delta_days)
        end = min(len(self), idx + self.delta_days + 1)
        return random.choice([i for i in range(start, end) if i != idx])

    def __del__(self):
        """Корректное завершение при удалении объекта"""
        self.io_executor.shutdown(wait=True)



# import pandas as pd
# import random
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from netCDF4 import Dataset as netCDF4_Dataset
# import os

# class Glorys12Dataset(torch.utils.data.Dataset):
#     def __init__(self, csv_file, transform1=None, transform2=None, random_seed=42, normalize=True, cache_size=64, delta_days = 15):
#         # Validate CSV file existence
#         if not os.path.exists(csv_file):
#             raise FileNotFoundError(f"CSV file {csv_file} not found")
            
#         # Read CSV file
#         self.data_frame = pd.read_csv(csv_file)
#         if len(self.data_frame) == 0:
#             raise ValueError("CSV file is empty")
        
#         self.file_paths = self.data_frame['File Path'].tolist()
        
        
#         self.transform1 = transform1
#         self.transform2 = transform2
#         self.delta_days = delta_days
#         self.variables = {
#             'mlotst': (0,),
#             'thetao': (0, 0),
#             'bottomT': (0,),
#             'uo': (0, 0),
#             'vo': (0, 0),
#             'so': (0, 0),
#             'zos': (0,)
#         }
        
#         if random_seed is not None:
#             random.seed(random_seed)
#             np.random.seed(random_seed)

#         self.cache = {}  # Cache for data loading
#         self.cache_size = cache_size

#     def __len__(self):
#         # Return the length of the dataset
#         return len(self.file_paths)

#     def __getitem__(self, idx: int):
#         # Load data for the given index
#         data_array = self._load_data_array(idx)
#  # здесь всегда положительная пара дней - не далее чем self.delta_days 5
        # if self.transform1 and self.transform2:
        #     data_aug_1 = self.transform1(data_array)
        #     data_aug_2 = self.transform2(self._load_data_array(self._random_close_idx(idx)))
#         return [data_aug_1, data_aug_2]
        


#     def _random_close_idx(self, idx):
#         # Generate a random index within delta_days from the current index
#         start_idx = max(0, idx - self.delta_days)
#         end_idx = min(len(self.data_frame), idx + self.delta_days + 1)
        
#         valid_indices = list(range(start_idx, end_idx))
        
#         # Remove the current index if it's in the valid range
#         if idx in valid_indices:
#             valid_indices.remove(idx)
        
#         if not valid_indices:
#             return idx
        
#         return random.choice(valid_indices)
#     def _load_data_array(self, idx):
#         # # Check cache first
#         # if idx in self.cache:
#         #     return self.cache[idx]

#         data_array = np.zeros((len(self.variables), 349, 661))
#         file_path = self.file_paths[idx] #self.data_frame.iloc[idx]['File Path']

#         try:
#             with netCDF4_Dataset(file_path, 'r') as ds:
#                 for i, (var_name, index) in enumerate(self.variables.items()):
#                     if var_name in ds.variables:
#                         variable_data = np.array(ds[var_name][index])
#                         variable_data = np.squeeze(variable_data)
#                         variable_data = np.where(variable_data == ds[var_name]._FillValue, 0, variable_data)
#                         data_array[i] = variable_data  # Save data to array
#                     else:
#                         print(f'Warning: Key {var_name} not found in dataset {file_path}')

#                 result = data_array.transpose((1, 2, 0))
#                 # self._update_cache(idx, result)  # Update the cache
#                 return result

#         except IndexError as e:
#             print(f'IndexError Exception while loading data from {file_path}: {e}')
#             raise e

#     def _update_cache(self, idx, data_array):
#         # Update cache with new data, maintaining cache size
#         if len(self.cache) >= self.cache_size:
#             self.cache.pop(next(iter(self.cache)))  # Remove oldest entry
#         self.cache[idx] = data_array  # Add new entry

















# import pandas as pd
# import random
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from netCDF4 import Dataset as netCDF4_Dataset

# class Glorys12Dataset(torch.utils.data.Dataset):
#     def __init__(self, csv_file, transform1=None, transform2=None, random_seed=42):
#         # Чтение CSV-файла
#         self.data_frame = pd.read_csv(csv_file)
#         self.transform1 = transform1
#         self.transform2 = transform2
#         self.variables = {
#             'mlotst': (0,),
#             'thetao': (0, 0),
#             'bottomT': (0,),
#             'uo': (0, 0),
#             'vo': (0, 0),
#             'so': (0, 0),
#             'zos': (0,)
#         }
        
#         if random_seed is not None:
#             random.seed(random_seed)
#             np.random.seed(random_seed)

#     def __len__(self):
#         return len(self.data_frame)

#     def __getitem__(self, idx):
#         data_array = self._load_data_array(idx)
#         if self.transform1 and self.transform2:
#             if random.random() > 0.5:
#                 data_aug_1 = self.transform1(data_array)
#                 data_aug_2 = self.transform2(data_array)
#                 data_aug = [data_aug_1, data_aug_2]
#             else:
#                 data_aug_1 = self.transform1(data_array)
#                 data_aug_2 = self.transform2(self._load_data_array(self._random_idx(idx)))
#                 data_aug = [data_aug_1, data_aug_2]
#             return data_aug
#         else:
#             return data_array


#     def _random_idx(self, idx):
#         # Определяем диапазоны
#         valid_indices = []
        
#         # Добавляем индексы от 0 до idx - 15 (если idx > 15)
#         if idx > 15:
#             valid_indices.extend(range(0, idx - 15))
        
#         # Добавляем индексы от idx + 15 до конца DataFrame
#         if idx < len(self.data_frame) - 15:
#             valid_indices.extend(range(idx + 15, len(self.data_frame)))

#         # Если нет доступных индексов, возвращаем текущий индекс
#         if not valid_indices:
#             return idx

#         # Выбираем случайный индекс из доступных
#         return random.choice(valid_indices)

#     def _load_data_array(self, idx):
#         data_array = np.zeros((len(self.variables), 349, 661))
#         file_path = self.data_frame.iloc[idx]['File Path']
#         try:
#             with netCDF4_Dataset(file_path, 'r') as ds:
#                 for i, (var_name, index) in enumerate(self.variables.items()):
#                     if var_name in ds.variables:
#                         variable_data = np.array(ds[var_name][index])
#                         variable_data = np.squeeze(variable_data)
#                         variable_data = np.where(variable_data == ds[var_name]._FillValue, 0, variable_data)
#                         data_array[i] = variable_data  # Сохраняем данные в массив
#                     else:
#                         print(f'Warning: Key {var_name} not found in dataset {file_path}')
#                 return data_array.transpose((1, 2, 0))
#         except IndexError as e:
#             print(f'IndexError Exception while loading data from {file_path}: {e}')
#             raise e
