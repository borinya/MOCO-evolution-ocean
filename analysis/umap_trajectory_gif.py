import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import umap
import imageio
from collections import defaultdict

def load_features(feature_dir):
    npy_files = sorted(glob.glob(os.path.join(feature_dir, '*.npy')))
    # Извлекаем дату из имени файла, например: '0_1993-01-01_features.npy' -> '1993-01-01'
    dates = [os.path.basename(f).split('_')[1] for f in npy_files]
    feats = [np.load(f) for f in npy_files]
    return np.array(feats), dates

def get_year_palette(years):
    """Вернёт dict: year -> цвет"""
    unique_years = sorted(set(years))
    cmap = plt.get_cmap('tab20', len(unique_years))  # до 20 лет — супер контрастно
    year2color = {year: cmap(i) for i, year in enumerate(unique_years)}
    return year2color

def make_umap_gif_by_year(feature_dir, out_dir='/app/MoCo/MOCOv3-MNIST/analysis', out_gif='umap_trajectory_by_year.gif', points_per_frame=30):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    feats, dates = load_features(feature_dir)
    years = [d[:4] for d in dates]
    year2color = get_year_palette(years)

    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(feats)

    frames = []
    n_points = len(dates)
    frame_idx = 0
    for i in range(points_per_frame, n_points+points_per_frame, points_per_frame):
        i = min(i, n_points)
        plt.figure(figsize=(8,6))
        # Рисуем трек раскрашенный по годам
        for j in range(1, i):
            y = years[j]
            plt.plot(emb[j-1:j+1,0], emb[j-1:j+1,1], color=year2color[y], lw=2)
        # Точки — тоже в цвет года
        for j in range(i):
            y = years[j]
            plt.scatter(emb[j,0], emb[j,1], color=year2color[y], s=20)
        # Подписываем последний год
        plt.text(emb[i-1,0], emb[i-1,1], dates[i-1], fontsize=8, color=year2color[years[i-1]], alpha=0.8)
        # Легенда по годам (если не слишком много)
        if len(year2color) <= 20:
            for y, c in year2color.items():
                plt.plot([], [], color=c, label=y)
            plt.legend(title='Year', fontsize=8, loc='best')
        plt.title(f'UMAP trajectory up to {dates[i-1]} ({i}/{n_points})')
        plt.xlabel('UMAP-1')
        plt.ylabel('UMAP-2')
        plt.tight_layout()
        fname = os.path.join(out_dir, f'umap_frame_{frame_idx:04d}.png')
        plt.savefig(fname)
        plt.close()
        frames.append(imageio.imread(fname))
        frame_idx += 1

    # Дублируем последний кадр в конце для паузы
    for _ in range(10):
        frames.append(frames[-1])

    out_gif_path = os.path.join(out_dir, out_gif)
    imageio.mimsave(out_gif_path, frames, duration=1.0)  # 1 секунда на каждый кадр
    print(f"UMAP trajectory gif saved to {out_gif_path}")

    # Чистим временные картинки
    for i in range(frame_idx):
        fname = os.path.join(out_dir, f'umap_frame_{i:04d}.png')
        if os.path.exists(fname):
            os.remove(fname)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, required=False, default='/app/MoCo/MOCOv3-MNIST/features_glorys12_moco256/2025-07-07_moco256_20250216_141630_checkpoint_0202')
    parser.add_argument('--out_dir', type=str, default='/app/MoCo/MOCOv3-MNIST/analysis')
    parser.add_argument('--out_gif', type=str, default='umap_trajectory_by_year.gif')
    parser.add_argument('--points_per_frame', type=int, default=20)
    args = parser.parse_args()
    make_umap_gif_by_year(args.feature_dir, args.out_dir, args.out_gif, args.points_per_frame)