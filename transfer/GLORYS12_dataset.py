import pandas as pd
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from netCDF4 import Dataset as netCDF4_Dataset
import os

class Glorys12Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform1=None, transform2=None, random_seed=42, normalize=True, cache_size=100):
        # Validate CSV file existence
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file {csv_file} not found")
            
        # Read CSV file
        self.data_frame = pd.read_csv(csv_file)
        if len(self.data_frame) == 0:
            raise ValueError("CSV file is empty")
        
        self.transform1 = transform1
        self.transform2 = transform2
        self.variables = {
            'mlotst': (0,),
            'thetao': (0, 0),
            'bottomT': (0,),
            'uo': (0, 0),
            'vo': (0, 0),
            'so': (0, 0),
            'zos': (0,)
        }

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.cache = {}  # Cache for data loading
        self.cache_size = cache_size

    def __len__(self):
        # Return the length of the dataset
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Load data for the given index
        data_array = self._load_data_array(idx)

        if self.transform1 and self.transform2:
            if random.random() > 0.5:
                data_aug_1 = self.transform1(data_array)
                data_aug_2 = self.transform2(data_array)
            else:
                data_aug_1 = self.transform1(data_array)
                data_aug_2 = self.transform2(self._load_data_array(self._random_idx(idx)))
            return [data_aug_1, data_aug_2]
        
        return data_array

    def _random_idx(self, idx):
        # Generate a random index that is not too close to the current index
        valid_indices = []
        if idx > 15:
            valid_indices.extend(range(0, idx - 15))
        if idx < len(self.data_frame) - 15:
            valid_indices.extend(range(idx + 15, len(self.data_frame)))

        if not valid_indices:
            return idx

        return random.choice(valid_indices)

    def _load_data_array(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]

        data_array = np.zeros((len(self.variables), 349, 661))
        file_path = self.data_frame.iloc[idx]['File Path']

        try:
            with netCDF4_Dataset(file_path, 'r') as ds:
                for i, (var_name, index) in enumerate(self.variables.items()):
                    if var_name in ds.variables:
                        variable_data = np.array(ds[var_name][index])
                        variable_data = np.squeeze(variable_data)
                        variable_data = np.where(variable_data == ds[var_name]._FillValue, 0, variable_data)
                        data_array[i] = variable_data  # Save data to array
                    else:
                        print(f'Warning: Key {var_name} not found in dataset {file_path}')

                result = data_array.transpose((1, 2, 0))
                self._update_cache(idx, result)  # Update the cache
                return result

        except IndexError as e:
            print(f'IndexError Exception while loading data from {file_path}: {e}')
            raise e

    def _update_cache(self, idx, data_array):
        # Update cache with new data, maintaining cache size
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))  # Remove oldest entry
        self.cache[idx] = data_array  # Add new entry

















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
