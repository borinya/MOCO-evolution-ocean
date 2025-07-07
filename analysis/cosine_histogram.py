import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm 
def load_features(feature_dir):
    npy_files = sorted(glob.glob(os.path.join(feature_dir, '*.npy')))
    # Извлекаем дату из имени файла, например: '0_1993-01-01_features.npy' -> '1993-01-01'
    dates = [os.path.basename(f).split('_')[1] for f in npy_files]
    feats = [np.load(f) for f in npy_files]
    return np.array(feats), dates

def cosine_histogram(feature_dir, out_png='cosine_histogram.png'):
    feats, dates = load_features(feature_dir)
    cosines = []
    
    for i in tqdm(range(1, len(feats))):
        sim = cosine_similarity(feats[i-1].reshape(1, -1), feats[i].reshape(1, -1))[0][0]
        cosines.append(sim)
    plt.figure(figsize=(8, 5))
    plt.hist(cosines, bins=40, color='tab:green', alpha=0.7)
    plt.xlabel('Cosine similarity between (i-1) and (i)')
    plt.ylabel('Count')
    plt.title('Cosine similarity histogram')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Cosine histogram saved to {out_png}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, required=False, default= '/app/MoCo/MOCOv3-MNIST/features_glorys12_moco256/2025-07-07_moco256_20250216_141630_checkpoint_0202')
    parser.add_argument('--out', type=str, default='cosine_histogram.png')
    args = parser.parse_args()
    cosine_histogram(args.feature_dir, args.out)