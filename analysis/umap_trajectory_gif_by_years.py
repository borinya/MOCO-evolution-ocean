import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import umap
import imageio

def load_features(feature_dir):
    npy_files = sorted(glob.glob(os.path.join(feature_dir, '*.npy')))
    # Извлекаем дату из имени файла, например: '0_1993-01-01_features.npy' -> '1993-01-01'
    dates = [os.path.basename(f).split('_')[1] for f in npy_files]
    feats = [np.load(f) for f in npy_files]
    return np.array(feats), dates

def make_gif_for_year(year, emb, dates, out_dir, points_per_frame=15):
    idxs = [i for i, d in enumerate(dates) if d.startswith(str(year))]
    if not idxs:
        print(f"No data for year {year}")
        return
    year_dir = os.path.join(out_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    frames = []
    n_points = len(idxs)
    frame_idx = 0
    for k in range(points_per_frame, n_points+points_per_frame, points_per_frame):
        k = min(k, n_points)
        plt.figure(figsize=(8,6))
        plt.plot(emb[idxs[:k],0], emb[idxs[:k],1], c='tab:blue', lw=2, marker='o', markersize=7, alpha=0.7)
        plt.scatter(emb[idxs[:k],0], emb[idxs[:k],1], c=np.linspace(0, 1, k), cmap='viridis', s=25)
        plt.text(emb[idxs[k-1],0], emb[idxs[k-1],1], dates[idxs[k-1]], fontsize=8, color='red', alpha=0.8)
        plt.title(f'UMAP trajectory for {year} up to {dates[idxs[k-1]]} ({k}/{n_points})')
        plt.xlabel('UMAP-1')
        plt.ylabel('UMAP-2')
        plt.tight_layout()
        fname = os.path.join(year_dir, f'umap_frame_{frame_idx:04d}.png')
        plt.savefig(fname)
        plt.close()
        frames.append(imageio.imread(fname))
        frame_idx += 1

    # Дублируем последний кадр для паузы
    for _ in range(10):
        frames.append(frames[-1])

    out_gif_path = os.path.join(year_dir, f'umap_trajectory_{year}.gif')
    imageio.mimsave(out_gif_path, frames, duration=1.0)
    print(f"UMAP trajectory gif for {year} saved to {out_gif_path}")

    # Чистим временные картинки
    for i in range(frame_idx):
        fname = os.path.join(year_dir, f'umap_frame_{i:04d}.png')
        if os.path.exists(fname):
            os.remove(fname)

def make_umap_gifs_by_years(feature_dir, out_dir='/app/MoCo/MOCOv3-MNIST/analysis/gifs', points_per_frame=15):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    feats, dates = load_features(feature_dir)
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(feats)
    years = sorted(set(d[:4] for d in dates))
    for year in years:
        make_gif_for_year(year, emb, dates, out_dir, points_per_frame)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, required=False, default='/app/MoCo/MOCOv3-MNIST/features_glorys12_moco256/2025-07-07_moco256_20250216_141630_checkpoint_0202')
    parser.add_argument('--out_dir', type=str, default='/app/MoCo/MOCOv3-MNIST/analysis/gifs')
    parser.add_argument('--points_per_frame', type=int, default=5)
    args = parser.parse_args()
    make_umap_gifs_by_years(args.feature_dir, args.out_dir, args.points_per_frame)