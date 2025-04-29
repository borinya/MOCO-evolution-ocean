#!/usr/bin/env python
import argparse
import os
import torch
import torch.optim as optim 
import torchvision.transforms as transforms

import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from moco.transformer_model import OceanTransformer
from moco.transformer_dataset import Glorys12SequenceDataset
from moco.builder import MoCo_ResNet
from torchvision.models import resnet50
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(description='Ocean State Transformer Forecasting')
    
    # Paths
    parser.add_argument('--csv-file', required=False, type=str, default='/app/MoCo/MOCOv3-MNIST/momental files and code/cleaned_data.csv', 
                      help='Path to cleaned data CSV')
    parser.add_argument('--checkpoint', required=False, type=str, default='/app/MoCo/MOCOv3-MNIST/checkpoints/20250404_124558_checkpoint_0299.pth.tar',
                      help='Path to pretrained MoCo checkpoint')
    
    # Model architecture
    parser.add_argument('--transformer-layers', type=int, default=4,
                      help='Number of transformer layers')
    parser.add_argument('--transformer-heads', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--transformer-dim-ff', type=int, default=1024,
                      help='Feedforward dimension')
    parser.add_argument('--transformer-dropout', type=float, default=0.1,
                      help='Transformer dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Base learning rate')
    parser.add_argument('--encoder-lr', type=float, default=1e-5,
                      help='Encoder learning rate if finetuning')
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--optimizer', choices=['adam', 'adamw'], default='adamw')
    parser.add_argument('--finetune-encoder', action='store_true',
                      help='Fine-tune the encoder')
    
    # Data parameters
    parser.add_argument('--seq-len', type=int, default=60,
                      help='Input sequence length')
    parser.add_argument('--pred-horizon', type=int, default=30,
                      help='Prediction horizon')
    parser.add_argument('--predict-differences', action='store_true',
                      help='Predict differences instead of absolute values')
    parser.add_argument('--transform', type=str, default=None,
                      help='Type of transformation to apply')
    parser.add_argument('--cache-size', type=int, default=512,
                      help='Dataset cache size')
    parser.add_argument('--num-io-workers', type=int, default=20,
                      help='Number of IO workers for dataset preprocessing')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                      help='Prefetch factor for data loading')
    parser.add_argument('--random-seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=0,
                      help='Number of workers for DataLoader')
    
    # Experiment management
    parser.add_argument('--amp', action='store_true',
                      help='Use mixed precision')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                      help='Gradient clipping norm')
    parser.add_argument('--log-dir', type=str, default='/app/MoCo/logs_transformer', #/app/MoCo/MOCOv3-MNIST/runs_transformer
                      help='Base directory for logs')
    parser.add_argument('--save-interval', type=int, default=10,
                      help='Checkpoint saving interval')
    
    return parser.parse_args()

def setup_logging(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"transformer_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    # Save all arguments to tensorboard
    for arg in vars(args):
        writer.add_text(f"args/{arg}", str(getattr(args, arg)))
        
    return writer, log_dir

def load_encoder(checkpoint_path, device, finetune=False):
    encoder = MoCo_ResNet(
        partial(resnet50, zero_init_residual=True), 
        dim=256, mlp_dim=4096, T=1.0
    ).base_encoder
    
    # Добавить weights_only=False для подавления предупреждения
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
    encoder.load_state_dict(state_dict, strict=False)
    
    if not finetune:
        for param in encoder.parameters():
            param.requires_grad = False
            
    return encoder.to(device)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logging and experiment tracking
    writer, log_dir = setup_logging(args)
    print(f"Experiment logs saved to: {log_dir}")
    
    # Initialize models
    encoder = load_encoder(args.checkpoint, device, args.finetune_encoder)

    transformer = OceanTransformer(
        input_dim=256,
        num_layers=args.transformer_layers,
        nhead=args.transformer_heads,
        dim_feedforward=args.transformer_dim_ff,
        dropout=args.transformer_dropout
    ).to(device)
    
    # Optimizer setup
    optim_params = [
        {'params': transformer.parameters(), 'lr': args.lr}
    ]
    if args.finetune_encoder:
        optim_params.append({'params': encoder.parameters(), 'lr': args.encoder_lr})
    
    optimizer = optim.AdamW(optim_params, weight_decay=args.weight_decay) if args.optimizer == 'adamw' \
        else optim.Adam(optim_params, weight_decay=args.weight_decay)
    
    # нормализация данных
    means = np.array([1.673302181686475, 33.37522164335293, 32.58433311325712, 
            11.152242330669477, 0.025353081653846376, -0.00907171541589713, 
            0.07366986763832623])
    
    square_means = np.array([5.995956099912317, 1720.1733657260818, 1063.4138676153, 
            149.60278359811343, 0.009805976106874816, 0.008356788111581723, 
            0.035208209865639856])
    stds = means**2 - square_means

    augmentation =  transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
])

    # Dataset and loader
    dataset = Glorys12SequenceDataset(
        csv_file=args.csv_file,
        sequence_length=args.seq_len,
        prediction_horizon=args.pred_horizon,
        predict_differences=args.predict_differences,
        transform=augmentation,
        cache_size=args.cache_size,
        num_io_workers=args.num_io_workers,
        prefetch_factor=args.prefetch_factor,
        random_seed=args.random_seed
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Training setup
    criterion = torch.nn.MSELoss()
    # scaler = GradScaler(enabled=args.amp)
    scaler = torch.amp.GradScaler(device_type='cuda', enabled=args.amp)
    global_step = 0
    
    for epoch in range(args.epochs):
        transformer.train()
        if args.finetune_encoder:
            encoder.train()
        else:
            encoder.eval()
            
        epoch_loss = 0.0
        
        for batch_idx, (sequences, targets) in enumerate(dataloader):
            sequences = sequences.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass through encoder
            with torch.set_grad_enabled(args.finetune_encoder):
                if args.amp:
                    with autocast():
                        batch_size, seq_len, H, W, C = sequences.shape
                        features = encoder(sequences.view(-1, H, W, C).view(batch_size, seq_len, -1))
                else:
                    batch_size, seq_len, H, W, C = sequences.shape
                    features = encoder(sequences.view(-1, H, W, C)).view(batch_size, seq_len, -1)
            
            # Transformer forward
            optimizer.zero_grad()
            
            if args.amp:
                with autocast():
                    predictions = transformer(features)
                    loss = criterion(predictions, targets)
            else:
                predictions = transformer(features)
                loss = criterion(predictions, targets)
            
            # Backward and optimize
            scaler.scale(loss).backward()
            
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.grad_clip)
                if args.finetune_encoder:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            epoch_loss += loss.item()
            global_step += 1
            writer.add_scalar('train/loss_step', loss.item(), global_step)
            
        # Epoch logging
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('train/loss_epoch', avg_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch+1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch+1,
                'transformer': transformer.state_dict(),
                'encoder': encoder.state_dict() if args.finetune_encoder else None,
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
                'loss': avg_loss
            }
            save_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    writer.close()

if __name__ == "__main__":
    main()
