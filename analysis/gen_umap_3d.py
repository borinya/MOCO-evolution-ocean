import os
import numpy as np
import glob
import umap
import plotly.graph_objs as go
import plotly.offline as py

def load_features(feature_dir):
    npy_files = sorted(glob.glob(os.path.join(feature_dir, '*.npy')))
    dates = [os.path.basename(f).split('_')[1] for f in npy_files]
    feats = [np.load(f) for f in npy_files]
    return np.array(feats), dates

def plot_umap_3d(feature_dir, out_html):
    feats, dates = load_features(feature_dir)
    years = [d[:4] for d in dates]
    unique_years = sorted(list(set(years)))
    year2int = {y: i for i, y in enumerate(unique_years)}
    colors = [year2int[y] for y in years]
    reducer = umap.UMAP(n_components=3, random_state=42)
    emb = reducer.fit_transform(feats)
    trace = go.Scatter3d(
        x=emb[:,0], y=emb[:,1], z=emb[:,2],
        mode='markers+lines',
        marker=dict(
            size=4,
            color=colors,
            colorscale='Viridis',
            colorbar=dict(title='Year', tickvals=list(year2int.values()), ticktext=unique_years)
        ),
        line=dict(
            color='rgba(50,50,50,0.2)',
            width=2
        ),
        text=dates
    )
    layout = go.Layout(
        title='3D UMAP Trajectory',
        scene=dict(
            xaxis_title='UMAP-1',
            yaxis_title='UMAP-2',
            zaxis_title='UMAP-3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig = go.Figure(data=[trace], layout=layout)
    py.plot(fig, filename=out_html, auto_open=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--out_html', type=str, required=True)
    args = parser.parse_args()
    plot_umap_3d(args.feature_dir, args.out_html) 