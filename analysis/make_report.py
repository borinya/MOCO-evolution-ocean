import os
import glob

def find_files_by_suffix(directory, suffix):
    return sorted(glob.glob(os.path.join(directory, '**', f'*{suffix}'), recursive=True))

def make_report(analysis_dir, report_path):
    with open(report_path, 'w') as f:
        f.write('<html><head><meta charset="utf-8"><title>MoCo Features Analysis Report</title></head><body>\n')
        f.write('<h1>MoCo Features Analysis Report</h1>\n')

        # Cosine histogram
        f.write('<h2>Cosine Similarity Histogram</h2>\n')
        cos_hist = find_files_by_suffix(analysis_dir, 'cosine_histogram.png')
        if cos_hist:
            f.write(f'<img src="{cos_hist[0]}" width="600"><br>\n')

        # Silhouette
        f.write('<h2>Clustering Quality (Silhouette Score)</h2>\n')
        sil_files = find_files_by_suffix(analysis_dir, 'silhouette.txt')
        if sil_files:
            with open(sil_files[0]) as txt:
                score = txt.read()
            f.write(f'<pre>{score}</pre>\n')

        # 2D UMAP GIFs by year
        f.write('<h2>UMAP Trajectory GIFs (by year)</h2>\n')
        gif_dirs = [d for d in glob.glob(os.path.join(analysis_dir, '*')) if os.path.isdir(d) and os.path.basename(d).isdigit()]
        for year_dir in sorted(gif_dirs):
            year = os.path.basename(year_dir)
            gifs = find_files_by_suffix(year_dir, '.gif')
            if gifs:
                f.write(f'<b>{year}</b>:<br><img src="{gifs[0]}" width="450"><br>\n')

        # 3D UMAP interactive
        f.write('<h2>3D UMAP Trajectory (Interactive)</h2>\n')
        htmls = find_files_by_suffix(analysis_dir, '.html')
        if htmls:
            f.write(f'<a href="{htmls[0]}">Open interactive 3D UMAP</a><br>\n')

        f.write('</body></html>\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis_dir', type=str, default='/app/MoCo/MOCOv3-MNIST/analysis')
    parser.add_argument('--report_path', type=str, default='/app/MoCo/MOCOv3-MNIST/analysis/report.html')
    args = parser.parse_args()
    make_report(args.analysis_dir, args.report_path)