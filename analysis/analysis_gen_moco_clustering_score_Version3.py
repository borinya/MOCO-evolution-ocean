import numpy as np
import glob
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_features(feature_dir):
    npy_files = sorted(glob.glob(os.path.join(feature_dir, '*.npy')))
    feats = [np.load(f) for f in npy_files]
    return np.array(feats)

def moco_silhouette(feature_dir, n_clusters=10, out_txt='silhouette.txt'):
    feats = load_features(feature_dir)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(feats)
    score = silhouette_score(feats, labels)
    with open(out_txt, 'w') as f:
        f.write(f'Silhouette score for KMeans({n_clusters}): {score:0.3f}\n')
    print(f'Silhouette score saved to {out_txt}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--out_txt', type=str, default='silhouette.txt')
    args = parser.parse_args()
    moco_silhouette(args.feature_dir, args.n_clusters, args.out_txt)