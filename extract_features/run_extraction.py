import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from extract_features.config import FEATURES_DIR, CSV_FILE, make_experiment_tag, MEANS, STDS
from extract_features.dataset import Glorys12Dataset
from extract_features.feature_extractor import load_checkpoint, get_features

from torchvision import transforms

def main():
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_tag = make_experiment_tag()
    output_dir = os.path.join(FEATURES_DIR, experiment_tag)
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEANS, std=STDS)
    ])

    dataset = Glorys12Dataset(
        csv_file=CSV_FILE,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    model, start_epoch = load_checkpoint(dataset.CHECKPOINT_PATH)
    model.to(device)

    for i, (images, dates) in enumerate(loader):
        images = images.float().to(device)
        features = get_features(model, images)
        for batch_idx in range(features.size(0)):
            idx_dataset = i * batch_size + batch_idx
            date_str = dates[batch_idx]
            filename = f"{idx_dataset}_{date_str.replace(' ', '_').replace(':', '-')}_features.npy"
            filepath = os.path.join(output_dir, filename)
            feature_vector = features[batch_idx].cpu().numpy()
            np.save(filepath, feature_vector)
        print(f"Processed batch {i} and saved feature vectors.")

if __name__ == "__main__":
    main()