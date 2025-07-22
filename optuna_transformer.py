import optuna
import subprocess
import argparse
import shlex
import os
import json
import time
from datetime import datetime

def objective(trial):
    # Параметры для оптимизации
    params = {
        'd_model': trial.suggest_categorical('d_model', [256, 512, 768]),
        'nhead': trial.suggest_categorical('nhead', [4, 8, 16]),
        'num_layers': trial.suggest_int('num_layers', 2, 6),
        'dim_feedforward': trial.suggest_categorical('dim_feedforward', [1024, 2048, 3072]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'lr_scheduler': trial.suggest_categorical('lr_scheduler', ['cosine', 'step', 'plateau']),
        'num_epochs': 100,  # Фиксируем 100 эпох на trial
        'early_stop_patience': 30,
    }

    # Собираем команду для запуска
    cmd = [
        'python', 'main_transformer_train.py',
        f'--d_model={params["d_model"]}',
        f'--nhead={params["nhead"]}',
        f'--num_layers={params["num_layers"]}',
        f'--dim_feedforward={params["dim_feedforward"]}',
        f'--dropout={params["dropout"]}',
        f'--learning_rate={params["learning_rate"]}',
        f'--batch_size={params["batch_size"]}',
        f'--lr_scheduler={params["lr_scheduler"]}',
        f'--num_epochs={params["num_epochs"]}',
        f'--early_stop_patience={params["early_stop_patience"]}',
        '--log_dir=/app/transformer/logs/optuna',
        '--model_dir=/app/transformer/models/optuna',
    ]

    # Запускаем обучение
    print(f"\n🚀 Запуск trial {trial.number} с параметрами:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )

    # Собираем вывод в реальном времени
    best_loss = None
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            # Ищем строку с лучшим лоссом
            if "Best: " in output:
                best_loss = float(output.split("Best: ")[1].split()[0])
    
    # Проверяем результат
    return_code = process.poll()
    if return_code != 0:
        print(f"❌ Ошибка в trial {trial.number} (код {return_code})")
        raise optuna.exceptions.TrialPruned()
    
    if best_loss is None:
        print(f"❌ Не удалось найти лучший лосс в trial {trial.number}")
        raise optuna.exceptions.TrialPruned()
    
    # Сохраняем параметры и результат
    trial.set_user_attr("params", params)
    trial.set_user_attr("best_loss", best_loss)
    
    return best_loss

if __name__ == "__main__":
    # Создаем исследование
    study = optuna.create_study(
        direction='minimize',
        study_name='transformer_optimization',
        storage='sqlite:///transformer_optuna.db',
        load_if_exists=True
    )

    # Запускаем оптимизацию
    study.optimize(
        objective, 
        n_trials=100,
        n_jobs=1,  # Запускаем последовательно из-за использования GPU
        show_progress_bar=True
    )

    # Выводим результаты
    print("\n" + "="*50)
    print("🏆 Лучшие параметры:")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    print(f"\n🎯 Лучший валидационный лосс: {study.best_value:.4f}")
    
    # Сохраняем результаты в JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"optuna_results_{timestamp}.json", "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
            "trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "user_attrs": t.user_attrs
                } for t in study.trials
            ]
        }, f, indent=2)