import os
import numpy as np
import glob
import umap
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
from tqdm import tqdm

def load_features(feature_dir, dataset_fraction=1.0):
    npy_files = sorted(glob.glob(os.path.join(feature_dir, '*.npy')))
    n_total = len(npy_files)
    n_keep = max(1, int(n_total * dataset_fraction))
    npy_files = npy_files[:n_keep]
    print(f"Found {n_total} npy files, using first {n_keep} files ({dataset_fraction*100:.1f}%)")
    dates = [os.path.basename(f).split('_')[1] for f in npy_files]
    feats = [np.load(f) for f in npy_files]
    return np.array(feats), dates

def get_year(date):
    return int(date[:4])

def get_contrasting_colors(n):
    cmaps = [plt.get_cmap('tab10'), plt.get_cmap('tab20'), plt.get_cmap('hsv')]
    if n <= 10:
        return [cmaps[0](i) for i in range(n)]
    elif n <= 20:
        return [cmaps[1](i) for i in range(n)]
    else:
        return [cmaps[2](i / n) for i in range(n)]

def make_3d_gif_recent_years(
    feature_dir, 
    out_gif='umap_3d_recent_years.gif', 
    fps=10, 
    years_window=3,
    dataset_fraction=0.2,
):
    feats, dates = load_features(feature_dir, dataset_fraction)
    years = [get_year(d) for d in dates]
    unique_years = sorted(set(years))
    reducer = umap.UMAP(n_components=3, random_state=42)
    emb = reducer.fit_transform(feats)
    
    frames = []
    num_points = len(feats)
    duration = 1.0 / fps

    print("Rendering frames:")
    for i in tqdm(range(years_window, num_points + 1)):
        current_year = years[i-1]
        window_years = [y for y in range(current_year - years_window + 1, current_year + 1)]
        idxs = [j for j in range(i) if years[j] in window_years]
        if not idxs:
            continue
        window_unique_years = sorted(set(years[j] for j in idxs))
        colors = get_contrasting_colors(len(window_unique_years))
        year2color = {y: colors[n] for n, y in enumerate(window_unique_years)}
        point_colors = [year2color[years[j]] for j in idxs]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for k in range(1, len(idxs)):
            y = years[idxs[k]]
            ax.plot(
                [emb[idxs[k-1],0], emb[idxs[k],0]],
                [emb[idxs[k-1],1], emb[idxs[k],1]],
                [emb[idxs[k-1],2], emb[idxs[k],2]],
                color=year2color[y], lw=2, alpha=0.8
            )
        ax.scatter(
            emb[idxs,0], emb[idxs,1], emb[idxs,2],
            c=point_colors, s=30, alpha=0.95, edgecolor='k', linewidth=0.3
        )
        ax.text(
            emb[idxs[-1],0], emb[idxs[-1],1], emb[idxs[-1],2],
            dates[idxs[-1]], color=year2color[years[idxs[-1]]], fontsize=9, weight='bold'
        )
        ax.set_title(f"UMAP 3D Trajectory: {window_years[0]}–{window_years[-1]} (до {dates[i-1]})")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_zlabel("UMAP-3")
        handles = [plt.Line2D([0],[0], color=year2color[y], lw=4, label=str(y)) for y in window_unique_years]
        ax.legend(handles=handles, title="Year", loc="upper left", fontsize=8)
        ax.view_init(elev=20, azim=30 + (i * 1.5) % 360)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)
    
    imageio.mimsave(out_gif, frames, fps=fps)
    print(f"GIF saved to {out_gif}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--out_gif', type=str, default='umap_3d_recent_years.gif')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--years_window', type=int, default=3)
    parser.add_argument('--dataset_fraction', type=float, default=0.2, help="Fraction of dataset to use (from start), e.g. 0.2 is first 20%")
    args = parser.parse_args()
    make_3d_gif_recent_years(
        args.feature_dir, 
        args.out_gif, 
        args.fps, 
        args.years_window,
        args.dataset_fraction,
    )
    
    
# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--feature_dir', type=str, required=False, default='/app/MoCo/MOCOv3-MNIST/features_glorys12_moco256/2025-07-07_moco256_20250216_141630_checkpoint_0202')
#     parser.add_argument('--out_dir', type=str, default='/app/MoCo/MOCOv3-MNIST/analysis')
#     parser.add_argument('--out_gif', type=str, default='umap_trajectory_by_year.gif')
#     parser.add_argument('--points_per_frame', type=int, default=20)
#     args = parser.parse_args()
#     make_umap_gif_by_year(args.feature_dir, args.out_dir, args.out_gif, args.points_per_frame)