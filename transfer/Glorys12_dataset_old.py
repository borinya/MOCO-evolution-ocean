import glob
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from netCDF4 import Dataset as netCDF4_Dataset


# class GaussianNoiseAugmentation:
#     def __init__(self, mean=0.0, std=0.01, noisesize=(10, 10)):
#         self.mean = mean
#         self.std = std
#         self.noisesize = noisesize

#     def __call__(self, tensor):
#         channels, height, width = tensor.shape
#         noise = np.random.normal(self.mean, self.std, (channels, *self.noisesize))
#         noisetensor = torch.from_numpy(noise).float()
#         noisetensor = transforms.functional.resize(noisetensor, (height, width))
#         return tensor + noisetensor


class Glorys12Dataset(torch.utils.data.Dataset):
    def __init__(self, train_path, transform1=None, transform2=None, random_seed=None):
        self.list_paths = sorted(glob.glob(train_path))
        self.transform1 = transform1
        self.transform2 = transform2
        self.features_maps = ['thetao', 'vo', 'uo', 'so', 'zos', 'bottomT']
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

    def __len__(self):
        return len(self.list_paths)

    def __getitem__(self, idx):
        data_array = self._load_data_array(idx)
        if self.transform1 and self.transform2:
            if random.random() > 0.5:
                data_aug_1 = self.transform1(data_array)
                data_aug_2 = self.transform2(data_array)
                data_aug = [data_aug_1, data_aug_2]
            else:
                data_array_another = self._load_data_array(self._random_idx(idx))
                data_aug_1 = self.transform1(data_array)
                data_aug_2 = self.transform2(data_array_another)
                data_aug = [data_aug_1, data_aug_2]
            return data_aug
        else:
            return data_array

    def _random_idx(self, idx):
        random_number_idx = random.randint(14, 28)
        new_idx = idx + random_number_idx if random_number_idx + idx < len(self.list_paths) else idx - random_number_idx
        return max(0, min(new_idx, len(self.list_paths) - 1))

    def _load_data_array(self, idx):
        data_array = np.zeros((2 * len(self.features_maps), 349, 661))
        try:
            with netCDF4_Dataset(self.list_paths[idx], 'r') as ds:
                # for i, key in enumerate(self.features_maps):
                for i, (var_name, index) in enumerate(variables.items()):
                    if var_name in ds.variables:
                        variable_data = np.array(ds[var_name][index])
                        variable_data = np.squeeze(variable_data)
                        variable_data = np.where(variable_data == ds[var_name]._FillValue, 0, variable_data)
                    else:
                        print(f'Warning: Key {var_name} not found in dataset {self.list_paths[idx]}')
                return data_array.transpose((1, 2, 0))
        except IndexError as e:
            print(f'IndexError Exception: idx={idx}, key={key}, array_shape={variable_data.shape}')
            raise e
