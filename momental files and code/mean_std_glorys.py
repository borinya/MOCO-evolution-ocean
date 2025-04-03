import glob
from netCDF4 import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm 

# Путь к файлам .nc
pathes = sorted(glob.glob('/mnt/hippocamp/DATA/GLORYS12/????/*.nc'))

# Список для хранения данных о файлах
data = []

# Определяем переменные для анализа
variables = {
    'mlotst': (0,),
    'thetao': (0, 0),
    'bottomT': (0,),
    'uo': (0, 0),
    'vo': (0, 0),
    'so': (0, 0),
    'zos': (0,)
}

# Инициализируем счётчик
iteration_count = 0
for path in tqdm(pathes):
    try:
        with Dataset(path) as test_glorys:
            # Инициализируем переменные
            time_dimension = test_glorys['time'].shape[0] if 'time' in test_glorys.variables else None
            
            # Проходим по всем переменным, указанным в variables
            for var_name, shape in variables.items():
                if var_name in test_glorys.variables:
                    data_array = test_glorys[var_name][:]
                    # Проверяем, есть ли измерение времени
                    if time_dimension is not None:
                        for t in range(time_dimension):
                            # Если размерность временного измерения больше 1, вычисляем среднее и средний квадрат
                            if time_dimension > 1:
                                mean_val = np.mean(data_array[t, ...]) if len(shape) > 1 else np.mean(data_array[t])
                                mean_square_val = np.mean(data_array[t, ...]**2) if len(shape) > 1 else np.mean(data_array[t]**2)
                                
                                data.append({
                                    'File Path': path,
                                    'Variable': var_name,
                                    'Time Dimension': t + 1,
                                    'Mean': mean_val,
                                    'Mean Square': mean_square_val
                                })
                            else:
                                mean_val = np.mean(data_array)  # среднее по всем данным
                                mean_square_val = np.mean(data_array**2)  # средний квадрат по всем данным
                                
                                data.append({
                                    'File Path': path,
                                    'Variable': var_name,
                                    'Time Dimension': 1,
                                    'Mean': mean_val,
                                    'Mean Square': mean_square_val
                                })
                                
        #             # Увеличиваем счётчик итераций
        # iteration_count += 1

        # # Проверяем, достигли ли мы 10 итераций
        # if iteration_count >= 10:
        #     print("Reached 10 iterations, stopping the loop.")
        # break
    except Exception as e:
        # Обработка ошибок (например, файл может быть поврежден или не поддерживается)
        print(f"Error processing {path}: {e}")

# Создаем DataFrame из списка
df = pd.DataFrame(data)

# Сохраняем DataFrame в CSV файл
df.to_csv('files_with_mean.csv', index=False)

# Выводим информацию о завершении
print("Data saved to 'files_with_statistics.csv'")
