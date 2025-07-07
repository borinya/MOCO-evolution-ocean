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
        cache_size=512,  # размер кэша
        num_io_workers=20,  # Количество параллельных IO workers
        prefetch_factor=2  # Предзагрузка следующих элементов
    ):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file {csv_file} not found")
            
        self.data_frame = pd.read_csv(csv_file)
        if len(self.data_frame) == 0:
            raise ValueError("CSV file is empty")

        self.file_paths = self.data_frame['File Path'].tolist()
        self.transform1 = transform1
        self.transform2 = transform2
        self.delta_days = delta_days
        self.cache_size = cache_size
        self.num_io_workers = num_io_workers
        self.read_lock = threading.Lock()

        # Переменные netCDF, которые будем читать
        self.variables = {
            'mlotst': (0,),
            'thetao': (0, 0),
            'bottomT': (0,),
            'uo': (0, 0),
            'vo': (0, 0),
            'so': (0, 0),
            'zos': (0,)
        }

        self.cache = {}
        self.cache_lock = threading.Lock()
        self.pending_futures = {}
        self.io_executor = ThreadPoolExecutor(max_workers=num_io_workers)
        self.prefetch_factor = prefetch_factor

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def __len__(self):
        return len(self.file_paths)

def __getitem__(self, idx):
    close_idx = self._random_close_idx(idx)
    self._prefetch_adjacent(idx)
    self._prefetch_adjacent(close_idx)
    data_array1 = self._get_cached_data(idx)
    data_array2 = self._get_cached_data(close_idx)
    # Получаем дату из датафрейма:
    date_str = str(self.data_frame.iloc[idx]['Date']) if 'Date' in self.data_frame.columns else ""
    if self.transform1 and self.transform2:
        data_array1 = self.transform1(data_array1)
        data_array2 = self.transform2(data_array2)
    return data_array1, date_str  # <-- теперь возвращаем дату!

    def _prefetch_adjacent(self, idx):
        for offset in range(1, self.prefetch_factor + 1):
            next_idx = idx + offset
            if next_idx < len(self):
                self._async_load(next_idx)

    def _async_load(self, idx):
        with self.cache_lock:
            if idx in self.cache or idx in self.pending_futures:
                return
        future = self.io_executor.submit(self._load_single_file, idx)
        self.pending_futures[idx] = future

    def _get_cached_data(self, idx):
        with self.cache_lock:
            if idx in self.cache:
                return self.cache[idx]
        if idx in self.pending_futures:
            data = self.pending_futures[idx].result()
            with self.cache_lock:
                self._update_cache(idx, data)
                del self.pending_futures[idx]
            return data
        return self._load_single_file(idx)

    def _load_single_file(self, idx):
        file_path = self.file_paths[idx]
        data_array = np.zeros((len(self.variables), 349, 661), dtype=np.float32)
        try:
            with self.read_lock:
                with netCDF4_Dataset(file_path, 'r') as ds:
                    for i, (var_name, index) in enumerate(self.variables.items()):
                        if var_name in ds.variables:
                            var = ds[var_name]
                            variable_data = np.array(var[index], dtype=np.float32)
                            variable_data = np.squeeze(variable_data)
                            # Обработка разных вариантов размерностей
                            if variable_data.shape == (349, 661):
                                data_array[i] = variable_data
                            elif variable_data.shape == (2, 349, 661):
                                data_array[i] = variable_data[0]
                                print(f"Warning: {file_path} {var_name} shape (2,349,661), used [0]")
                            else:
                                raise ValueError(f"Unexpected shape {variable_data.shape} for {var_name} in {file_path}")
                            # Обработка nan-значений
                            if hasattr(var, '_FillValue'):
                                data_array[i] = np.where(data_array[i] == var._FillValue, np.nan, data_array[i])
                        else:
                            # Если переменной нет в файле - оставить как есть (нули)
                            print(f"Warning: {file_path} does not have variable {var_name}")
                    # Транспонируем в (349,661,7) для совместимости с torch
                    result = data_array.transpose((1, 2, 0))
                    result = np.nan_to_num(result, nan=0.0)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise
        with self.cache_lock:
            self._update_cache(idx, result)
        return result

    def _update_cache(self, idx, data):
        if len(self.cache) >= self.cache_size:
            del self.cache[next(iter(self.cache))]
        self.cache[idx] = data

    def _random_close_idx(self, idx):
        start = max(0, idx - self.delta_days)
        end = min(len(self), idx + self.delta_days + 1)
        return random.choice([i for i in range(start, end) if i != idx])

    def __del__(self):
        self.io_executor.shutdown(wait=True)

# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset
# from netCDF4 import Dataset as NetCDFDataset

# class Glorys12Dataset(Dataset):
#     def __init__(self, csv_file, transform=None):
#         self.data_frame = pd.read_csv(csv_file)
#         self.file_paths = self.data_frame['File Path'].tolist()
#         self.datetimes = self.data_frame['Datetime'].tolist()
#         self.transform = transform

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         file_path = self.file_paths[idx]
#         arr = np.zeros((7, 349, 661), dtype=np.float32)
#         try:
#             nc = NetCDFDataset(file_path)
#             for i, var in enumerate(['mlotst', 'thetao', 'bottomT', 'uo', 'vo', 'so', 'zos']):
#                 arr[i] = nc.variables[var][:]
#             nc.close()
#         except Exception as e:
#             print(f"Ошибка чтения {file_path}: {e}")
#             arr[:] = np.nan
#         if self.transform:
#             arr = self.transform(arr)
#         date = self.datetimes[idx]
#         return arr, date