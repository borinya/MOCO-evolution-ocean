import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import umap

def load_features(feature_dir):
    npy_files = sorted(glob.glob(os.path.join(feature_dir, '*.npy')))
    dates = [os.path.splitext(os.path.basename(f))[0] for f in npy_files]
    feats = [np.load(f) for f in npy_files]
    return np.array(feats), dates

def make_umap_trajectory_plot(feature_dir, out_png='umap_trajectory.png'):
    feats, dates = load_features(feature_dir)
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(feats)
    plt.figure(figsize=(10, 7))
    plt.plot(emb[:, 0], emb[:, 1], marker='o', lw=1, c='tab:blue')
    plt.scatter(emb[:, 0], emb[:, 1], c=range(len(dates)), cmap='viridis', s=15)
    for i in range(0, len(dates), max(1, len(dates)//20)):
        plt.text(emb[i,0], emb[i,1], dates[i], fontsize=7, alpha=0.7)
    plt.colorbar(label='time index')
    plt.title('UMAP Trajectory of GLORYS12 features')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"UMAP trajectory plot saved to {out_png}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--out', type=str, default='umap_trajectory.png')
    args = parser.parse_args()
    make_umap_trajectory_plot(args.feature_dir, args.out)