"""
Visualize 3D SDF (Signed Distance Function) for training samples.

This script:
1. Loads EM images and instance masks
2. Computes 3D SDF from the masks
3. Saves visualizations showing:
   - Original EM images
   - Instance masks (colored)
   - SDF maps (with colormap: blue=negative/inside, red=positive/outside, white=boundary)

Usage:
    conda activate agq
    python scripts/visualize_sdf.py --dataset em --num_samples 2 --out_dir outputs/visualize_sdf
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

import torch
import torchvision.utils as vutils

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectomics.config import get_cfg_defaults
from projects.AGQ.config import add_custom_config
from connectomics.data.augmentation import build_train_augmentor
from connectomics.data.dataset import build_dataloader
from connectomics.data.utils import compute_sdf_3d_fast


def get_sdf_colormap():
    """
    Create a diverging colormap for SDF visualization.
    Blue (negative/inside) -> White (boundary/zero) -> Red (positive/outside)
    """
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
              '#ffffff',  # white at zero
              '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    return mcolors.LinearSegmentedColormap.from_list('sdf', colors, N=256)


def colorize_instances(mask, max_instances=100):
    """
    Colorize instance mask with random colors.
    
    Args:
        mask: Instance mask (D, H, W) with integer labels
        max_instances: Maximum number of instances for color lookup
    
    Returns:
        RGB colored mask (D, H, W, 3)
    """
    # Create random color lookup table
    np.random.seed(42)  # For reproducibility
    colors = np.random.rand(max_instances + 1, 3)
    colors[0] = [0, 0, 0]  # Background is black
    
    # Map instance IDs to colors
    mask_flat = mask.flatten().astype(int) % (max_instances + 1)
    colored = colors[mask_flat].reshape(mask.shape + (3,))
    
    return colored


def visualize_sdf_slice(image, mask, sdf, slice_idx, out_path, sdf_cmap=None):
    """
    Visualize a single slice showing image, mask, and SDF.
    
    Args:
        image: Image volume (D, H, W)
        mask: Instance mask (D, H, W)
        sdf: SDF volume (D, H, W)
        slice_idx: Z-slice index to visualize
        out_path: Output file path
        sdf_cmap: Colormap for SDF
    """
    if sdf_cmap is None:
        sdf_cmap = get_sdf_colormap()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Image
    axes[0].imshow(image[slice_idx], cmap='gray')
    axes[0].set_title(f'EM Image (z={slice_idx})')
    axes[0].axis('off')
    
    # Instance mask (colored)
    colored_mask = colorize_instances(mask)
    axes[1].imshow(colored_mask[slice_idx])
    axes[1].set_title('Instance Mask')
    axes[1].axis('off')
    
    # SDF with colormap
    sdf_slice = sdf[slice_idx]
    vmax = max(abs(sdf_slice.min()), abs(sdf_slice.max()))
    im = axes[2].imshow(sdf_slice, cmap=sdf_cmap, vmin=-vmax, vmax=vmax)
    axes[2].set_title('SDF (blue=inside, red=outside)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    # SDF overlaid on image
    axes[3].imshow(image[slice_idx], cmap='gray', alpha=0.7)
    im_overlay = axes[3].imshow(sdf_slice, cmap=sdf_cmap, alpha=0.5, vmin=-vmax, vmax=vmax)
    axes[3].set_title('SDF Overlay')
    axes[3].axis('off')
    plt.colorbar(im_overlay, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_sdf_grid(image, mask, sdf, out_path, num_slices=8, sdf_cmap=None):
    """
    Visualize multiple slices in a grid.
    
    Args:
        image: Image volume (D, H, W)
        mask: Instance mask (D, H, W)
        sdf: SDF volume (D, H, W)
        out_path: Output file path
        num_slices: Number of slices to show
        sdf_cmap: Colormap for SDF
    """
    if sdf_cmap is None:
        sdf_cmap = get_sdf_colormap()
    
    D = image.shape[0]
    slice_indices = np.linspace(0, D-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(4, num_slices, figsize=(3*num_slices, 12))
    
    vmax = max(abs(sdf.min()), abs(sdf.max()))
    colored_mask = colorize_instances(mask)
    
    for i, z in enumerate(slice_indices):
        # Row 0: EM Image
        axes[0, i].imshow(image[z], cmap='gray')
        axes[0, i].set_title(f'z={z}', fontsize=10)
        axes[0, i].axis('off')
        
        # Row 1: Instance mask
        axes[1, i].imshow(colored_mask[z])
        axes[1, i].axis('off')
        
        # Row 2: SDF
        axes[2, i].imshow(sdf[z], cmap=sdf_cmap, vmin=-vmax, vmax=vmax)
        axes[2, i].axis('off')
        
        # Row 3: Overlay
        axes[3, i].imshow(image[z], cmap='gray', alpha=0.7)
        axes[3, i].imshow(sdf[z], cmap=sdf_cmap, alpha=0.5, vmin=-vmax, vmax=vmax)
        axes[3, i].axis('off')
    
    # Row labels
    axes[0, 0].set_ylabel('Image', fontsize=12)
    axes[1, 0].set_ylabel('Instances', fontsize=12)
    axes[2, 0].set_ylabel('SDF', fontsize=12)
    axes[3, 0].set_ylabel('Overlay', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def build_cfg_and_dataloader(dataset_name: str, num_workers: int = 4):
    """Build config and dataloader for a dataset."""
    if dataset_name.lower() in ['em', 'our']:
        config_base = 'projects/AGQ/configs/EM-Base.yaml'
        config_file = 'projects/AGQ/configs/AGQ-EM.yaml'
    elif dataset_name.lower() in ['ac3', 'ac3-ac4', 'ac3_ac4']:
        config_base = 'projects/AGQ/configs/SNEMI-Base.yaml'
        config_file = 'projects/AGQ/configs/AGQ.yaml'
    else:
        raise ValueError('Unsupported dataset: ' + dataset_name)

    args = type('Args', (), {})()
    args.config_base = config_base
    args.config_file = config_file
    args.distributed = False
    args.local_rank = None
    args.inference = False
    args.checkpoint = None
    args.opts = []

    cfg = get_cfg_defaults()
    add_custom_config(cfg)
    cfg.merge_from_file(args.config_base)
    cfg.merge_from_file(args.config_file)

    cfg.defrost()
    cfg.SYSTEM.NUM_CPUS = num_workers
    cfg.SOLVER.SAMPLES_PER_BATCH = 1
    cfg.freeze()

    augmentor = build_train_augmentor(cfg)
    dataloader = build_dataloader(
        cfg, augmentor, mode='train', rank=None,
        dataset_options={'ensure_single_connected': True, 'return_clean_input': True}
    )
    return cfg, dataloader


def visualize_dataset_sdf(dataset_name: str, num_samples: int, out_dir: str, truncate: float = 10.0):
    """
    Visualize SDF for dataset samples.
    
    Args:
        dataset_name: 'em' or 'ac3'
        num_samples: Number of samples to visualize
        out_dir: Output directory
        truncate: SDF truncation distance
    """
    cfg, dataloader = build_cfg_and_dataloader(dataset_name)
    
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    sdf_cmap = get_sdf_colormap()
    
    for batch_idx, sample in enumerate(dataloader):
        if batch_idx >= num_samples:
            break
        
        # Get image and label from sample
        # sample.out_input: (B, C, D, H, W)
        # sample.out_target_l[0]: instance labels (B, D, H, W) for target option "9-1"
        
        image = sample.out_input.cpu().numpy()  # (B, C, D, H, W)
        
        # Get the instance label (first target which is "9-1" in AGQ config)
        label = sample.out_target_l[0].cpu().numpy()  # (B, D, H, W) or (B, C, D, H, W)
        
        for b in range(image.shape[0]):
            img_vol = image[b, 0]  # (D, H, W)
            
            # Handle label shape
            if label.ndim == 5:  # (B, C, D, H, W)
                lbl_vol = label[b, 0]
            else:  # (B, D, H, W)
                lbl_vol = label[b]
            
            # Denormalize image if needed
            img_vol = (img_vol * cfg.DATASET.STD + cfg.DATASET.MEAN) * 255
            img_vol = np.clip(img_vol, 0, 255).astype(np.uint8)
            
            # Convert label to int for SDF computation
            lbl_vol = lbl_vol.astype(np.int32)
            
            # Compute SDF
            print(f"Computing SDF for batch {batch_idx}, sample {b}...")
            print(f"  Image shape: {img_vol.shape}, Label shape: {lbl_vol.shape}")
            print(f"  Unique labels: {len(np.unique(lbl_vol))}")
            
            sdf_vol = compute_sdf_3d_fast(lbl_vol, truncate_distance=truncate)
            
            print(f"  SDF range: [{sdf_vol.min():.2f}, {sdf_vol.max():.2f}]")
            
            # Save grid visualization
            grid_path = os.path.join(out_dir, f'{dataset_name}_batch{batch_idx}_sample{b}_grid.png')
            visualize_sdf_grid(img_vol, lbl_vol, sdf_vol, grid_path, num_slices=8, sdf_cmap=sdf_cmap)
            print(f"  Saved grid: {grid_path}")
            
            # Save individual slices (middle slice)
            mid_z = img_vol.shape[0] // 2
            slice_path = os.path.join(out_dir, f'{dataset_name}_batch{batch_idx}_sample{b}_z{mid_z}.png')
            visualize_sdf_slice(img_vol, lbl_vol, sdf_vol, mid_z, slice_path, sdf_cmap=sdf_cmap)
            print(f"  Saved slice: {slice_path}")
    
    print(f"\nVisualization complete! Results saved to: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize 3D SDF for training samples')
    parser.add_argument('--dataset', type=str, default='em', choices=['em', 'ac3'],
                        help='Dataset to visualize')
    parser.add_argument('--num_samples', type=int, default=2,
                        help='Number of samples to visualize')
    parser.add_argument('--out_dir', type=str, default='outputs/visualize_sdf',
                        help='Output directory')
    parser.add_argument('--truncate', type=float, default=10.0,
                        help='SDF truncation distance')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    visualize_dataset_sdf(
        args.dataset,
        args.num_samples,
        args.out_dir,
        args.truncate
    )
