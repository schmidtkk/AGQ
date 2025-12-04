"""
Training script for SDF prediction model using MONAI UNet3D.

This is an independent model from AGQ that predicts 3D Signed Distance Function (SDF)
from EM images.

Usage:
    conda activate agq
    python projects/AGQ/train_sdf.py --config projects/AGQ/configs/SDF-EM.yaml
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from connectomics.config import get_cfg_defaults
from connectomics.data.augmentation import build_train_augmentor
from connectomics.data.dataset import build_dataloader
from connectomics.engine.solver import build_optimizer, build_lr_scheduler

from projects.AGQ.config import add_custom_config

# Import model directly from file to avoid model/__init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "sdf_unet", 
    os.path.join(project_root, "projects/AGQ/model/sdf_unet.py")
)
sdf_unet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdf_unet_module)
SDFUNet3D = sdf_unet_module.SDFUNet3D
CombinedSDFLoss = sdf_unet_module.CombinedSDFLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train SDF prediction model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (overrides config)')
    return parser.parse_args()


def load_config(config_path: str):
    """Load and merge config files."""
    cfg = get_cfg_defaults()
    add_custom_config(cfg)
    
    # Add SDF-specific config options
    cfg.defrost()
    cfg.MODEL.SDF_CHANNELS = [32, 64, 128, 256]
    cfg.MODEL.SDF_STRIDES = [2, 2, 2]
    cfg.MODEL.SDF_NORM = 'instance'
    cfg.MODEL.SDF_DROPOUT = 0.0
    cfg.MODEL.SDF_TRUNCATE = 10.0
    cfg.MODEL.SDF_BOUNDARY_WEIGHT = 2.0
    cfg.MODEL.SDF_GRADIENT_WEIGHT = 0.1
    cfg.freeze()
    
    cfg.merge_from_file(config_path)
    return cfg


def build_model(cfg, device):
    """Build SDF prediction model."""
    # Parse channels from config
    channels = cfg.MODEL.FILTERS if hasattr(cfg.MODEL, 'FILTERS') else [32, 64, 128, 256]
    if len(channels) < 4:
        channels = [32, 64, 128, 256]
    
    strides = [2] * (len(channels) - 1)
    
    model = SDFUNet3D(
        in_channels=cfg.MODEL.IN_PLANES,
        out_channels=cfg.MODEL.OUT_PLANES,
        channels=tuple(channels),
        strides=tuple(strides),
        num_res_units=2,
        norm=cfg.MODEL.NORM_MODE if cfg.MODEL.NORM_MODE in ['batch', 'instance', 'group'] else 'instance',
        dropout=0.0,
    )
    
    model = model.to(device)
    return model


def build_criterion(cfg):
    """Build loss function."""
    return CombinedSDFLoss(
        l1_weight=1.0,
        gradient_weight=0.1,
        boundary_weight=2.0,
    )


class SDFTrainer:
    """Trainer class for SDF prediction model."""
    
    def __init__(self, cfg, device, checkpoint=None):
        self.cfg = cfg
        self.device = device
        
        # Build model
        self.model = build_model(cfg, device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        # Build optimizer and scheduler
        self.optimizer = build_optimizer(cfg, self.model)
        self.lr_scheduler = build_lr_scheduler(cfg, self.optimizer)
        
        # Build loss
        self.criterion = build_criterion(cfg)
        
        # Mixed precision
        self.scaler = GradScaler() if cfg.MODEL.MIXED_PRECESION else None
        
        # Build dataloaders
        self.augmentor = build_train_augmentor(cfg)
        self.train_loader = build_dataloader(
            cfg, self.augmentor, mode='train', rank=None,
            dataset_options={'ensure_single_connected': False}
        )
        
        if cfg.DATASET.VAL_IMAGE_NAME is not None:
            self.val_loader = build_dataloader(
                cfg, None, mode='val', rank=None,
                dataset_options={'ensure_single_connected': False}
            )
        else:
            self.val_loader = None
        
        # Setup output directory
        self.output_dir = cfg.DATASET.OUTPUT_PATH
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'logs'))
        
        # Training state
        self.start_iter = 0
        self.best_val_loss = float('inf')
        
        # Load checkpoint if provided
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
    
    def load_checkpoint(self, path):
        """Load checkpoint."""
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'iteration' in checkpoint:
            self.start_iter = checkpoint['iteration']
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from iteration {self.start_iter}")
    
    def save_checkpoint(self, iteration, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iteration': iteration,
            'best_val_loss': self.best_val_loss,
        }
        path = os.path.join(self.output_dir, f'checkpoint_{iteration:06d}.pth.tar')
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth.tar')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def train(self):
        """Main training loop."""
        self.model.train()
        
        total_iters = self.cfg.SOLVER.ITERATION_TOTAL
        train_iter = iter(self.train_loader)
        
        total_time = 0
        
        for iteration in range(self.start_iter, total_iters):
            start_time = time.perf_counter()
            
            # Get batch
            try:
                sample = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                sample = next(train_iter)
            
            # Move to device
            inputs = sample.out_input.to(self.device, non_blocking=True)
            targets = sample.out_target_l[0].to(self.device, non_blocking=True)
            
            # Ensure targets have channel dimension
            if targets.ndim == 4:  # (B, D, H, W)
                targets = targets.unsqueeze(1)  # (B, 1, D, H, W)
            
            # Normalize targets to [-1, 1] for tanh output
            # SDF is already in [-truncate, truncate], normalize by truncate
            truncate = 10.0  # Should match config
            targets = targets / truncate
            targets = torch.clamp(targets, -1, 1)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(inputs)
                    loss, loss_dict = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss, loss_dict = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            # Update learning rate
            self.lr_scheduler.step()
            
            # Timing
            iter_time = time.perf_counter() - start_time
            total_time += iter_time
            
            # Logging
            if (iteration + 1) % self.cfg.MONITOR.ITERATION_NUM[0] == 0:
                avg_time = total_time / (iteration + 1 - self.start_iter)
                remaining = avg_time * (total_iters - iteration - 1) / 3600
                
                lr = self.optimizer.param_groups[0]['lr']
                print(f"[Iter {iteration+1:06d}] Loss: {loss.item():.4f} | "
                      f"L1: {loss_dict['sdf_l1']:.4f} | Grad: {loss_dict['sdf_grad']:.4f} | "
                      f"LR: {lr:.6f} | Time: {iter_time:.3f}s | ETA: {remaining:.2f}h")
                
                # Tensorboard
                self.writer.add_scalar('Loss/total', loss.item(), iteration)
                self.writer.add_scalar('Loss/l1', loss_dict['sdf_l1'], iteration)
                self.writer.add_scalar('Loss/gradient', loss_dict['sdf_grad'], iteration)
                self.writer.add_scalar('LR', lr, iteration)
            
            # Visualization
            if (iteration + 1) % self.cfg.MONITOR.ITERATION_NUM[1] == 0:
                self._visualize(inputs, targets, outputs, iteration)
            
            # Validation
            if self.val_loader is not None and (iteration + 1) % self.cfg.SOLVER.ITERATION_VAL == 0:
                self.validate(iteration)
            
            # Save checkpoint
            if (iteration + 1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
                self.save_checkpoint(iteration + 1)
        
        # Final save
        self.save_checkpoint(total_iters)
        self.writer.close()
        print("Training complete!")
    
    def validate(self, iteration):
        """Run validation."""
        self.model.eval()
        
        val_losses = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for sample in self.val_loader:
                inputs = sample.out_input.to(self.device)
                targets = sample.out_target_l[0].to(self.device)
                
                if targets.ndim == 4:
                    targets = targets.unsqueeze(1)
                
                targets = targets / 10.0  # Normalize
                targets = torch.clamp(targets, -1, 1)
                
                outputs = self.model(inputs)
                loss, loss_dict = self.criterion(outputs, targets)
                
                val_losses['total'] += loss.item()
                for k, v in loss_dict.items():
                    val_losses[k] += v
                num_batches += 1
        
        # Average
        for k in val_losses:
            val_losses[k] /= num_batches
        
        print(f"[Validation] Loss: {val_losses['total']:.4f} | "
              f"L1: {val_losses['sdf_l1']:.4f} | Grad: {val_losses['sdf_grad']:.4f}")
        
        # Tensorboard
        self.writer.add_scalar('Val/Loss', val_losses['total'], iteration)
        self.writer.add_scalar('Val/L1', val_losses['sdf_l1'], iteration)
        
        # Save best model
        if val_losses['total'] < self.best_val_loss:
            self.best_val_loss = val_losses['total']
            self.save_checkpoint(iteration + 1, is_best=True)
        
        self.model.train()
    
    def _visualize(self, inputs, targets, outputs, iteration):
        """Visualize predictions in tensorboard."""
        # Take middle slice
        z_mid = inputs.shape[2] // 2
        
        # Input image
        img = inputs[0, 0, z_mid].cpu()
        self.writer.add_image('Input', img.unsqueeze(0), iteration)
        
        # Target SDF
        tgt = targets[0, 0, z_mid].cpu()
        tgt_vis = (tgt + 1) / 2  # Normalize to [0, 1] for visualization
        self.writer.add_image('Target_SDF', tgt_vis.unsqueeze(0), iteration)
        
        # Predicted SDF
        pred = outputs[0, 0, z_mid].detach().cpu()
        pred_vis = (pred + 1) / 2
        self.writer.add_image('Pred_SDF', pred_vis.unsqueeze(0), iteration)
        
        # Error map
        error = torch.abs(pred - tgt)
        self.writer.add_image('Error', error.unsqueeze(0), iteration)


def main():
    args = parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Override output directory if specified
    if args.output is not None:
        cfg.defrost()
        cfg.DATASET.OUTPUT_PATH = args.output
        cfg.freeze()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = SDFTrainer(cfg, device, checkpoint=args.checkpoint)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
