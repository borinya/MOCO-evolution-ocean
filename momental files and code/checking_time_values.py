import glob
from netCDF4 import Dataset
import pandas as pd
from tqdm import tqdm 

# Путь к файлам .nc
pathes = sorted(glob.glob('/mnt/hippocamp/DATA/GLORYS12/????/*.nc'))

# Список для хранения данных о файлах
data = []

for path in tqdm(pathes):
    try:
        with Dataset(path) as test_glorys:
            # Инициализируем переменные
            time_dimension = None
            latitude_dimension = None
            longitude_dimension = None
            variables = test_glorys.variables
            # Проверяем размерность переменной 'time'
            if 'time' in test_glorys.variables:
                time_dimension = test_glorys['time'].shape[0]

            # # Проверяем размерности пространственных переменных
            # if 'latitude' in test_glorys.variables:
            #     latitude_dimension = test_glorys['latitude'].shape[0]
            # if 'longitude' in test_glorys.variables:
            #     longitude_dimension = test_glorys['longitude'].shape[0]

            # # Если временное измерение больше 1, добавляем информацию о файле в список
            
            data.append({
                'File Path': path,
                'Number of Time Dimensions': time_dimension,
                'Latitude Dimension': latitude_dimension,
                'Longitude Dimension': longitude_dimension,
                'variables':variables.keys()
            })
            if time_dimension and time_dimension > 1:
            data.append({
                'File Path': path,
                'Number of Time Dimensions': time_dimension,
            })  
    except Exception as e:
        # Обработка ошибок (например, файл может быть поврежден или не поддерживается)
        print(f"Error processing {path}: {e}")

# Создаем DataFrame из списка
df = pd.DataFrame(data)

# Сохраняем DataFrame в CSV файл
df.to_csv('files_with_dimensions.csv', index=False)

# Выводим информацию о завершении
print("Data saved to 'files_with_dimensions.csv'")


variables = {
    'mlotst': (0,),
    'thetao': (0, 0),
    'bottomT': (0,),
    'uo': (0, 0),
    'vo': (0, 0),
    'so': (0, 0),
    'zos': (0,)
}

# import glob
# from netCDF4 import Dataset
# import pandas as pd
# from tqdm import tqdm 

# # Путь к файлам .nc
# pathes = sorted(glob.glob('/mnt/hippocamp/DATA/GLORYS12/????/*.nc'))

# # Список для хранения данных о файлах
# data = []

# for path in tqdm(pathes):
#     test_glorys = Dataset(path)
    
#     # Проверяем размерность переменной 'time'
#     if 'time' in test_glorys.variables:
#         time_dimension = test_glorys['time'].shape[0]
#         # if time_dimension > 1:
#             # Добавляем информацию о файле в список
#         data.append({'File Path': path, 'Number of Time Dimensions': time_dimension})

# # Создаем DataFrame из списка
# df = pd.DataFrame(data)
# df.to_csv('files_with_dimensions.csv', index=False)
