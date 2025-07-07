import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from netCDF4 import Dataset as NetCDFDataset

class Glorys12Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.file_paths = self.data_frame['File Path'].tolist()
        self.datetimes = self.data_frame['Datetime'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        arr = np.zeros((7, 349, 661), dtype=np.float32)
        try:
            nc = NetCDFDataset(file_path)
            for i, var in enumerate(['mlotst', 'thetao', 'bottomT', 'uo', 'vo', 'so', 'zos']):
                arr[i] = nc.variables[var][:]
            nc.close()
        except Exception as e:
            print(f"Ошибка чтения {file_path}: {e}")
            arr[:] = np.nan
        if self.transform:
            arr = self.transform(arr)
        date = self.datetimes[idx]
        return arr, date