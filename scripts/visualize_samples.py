"""
Visualize training samples for AC3-AC4 and EM datasets.

Usage (examples):

# Visualize EM dataset samples (AGQ-EM config)
conda activate agq
python scripts/visualize_samples.py --dataset em --num_batches 3 --out_dir outputs/visualize_em

# Visualize AC3-AC4 samples with existing AGQ config
python scripts/visualize_samples.py --dataset ac3 --num_batches 3 --out_dir outputs/visualize_ac3

This script:
- Loads the proper configs for the dataset
- Builds train augmentor and dataloader
- Samples a few batches and saves images of the sampled volumes and labels as grids (png)
"""

import os
import argparse
import numpy as np
from pathlib import Path

import torch
import torchvision.utils as vutils

from connectomics.config import load_cfg, get_cfg_defaults
from projects.AGQ.config import add_custom_config
from connectomics.data.augmentation import build_train_augmentor
from connectomics.data.dataset import build_dataloader
from connectomics.utils.visualizer import Visualizer


def build_cfg_and_dataloader(dataset_name: str, num_workers: int = 4):
    # Select config base and config files depending on dataset
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

    # Ensure small test for debug
    # But do not override user-specified sizes.
    cfg.defrost()
    cfg.SYSTEM.NUM_CPUS = num_workers
    cfg.SOLVER.SAMPLES_PER_BATCH = 1
    cfg.freeze()

    augmentor = build_train_augmentor(cfg)
    dataloader = build_dataloader(cfg, augmentor, mode='train', rank=None,
                                  dataset_options={'ensure_single_connected': True,
                                                   'return_clean_input': True})
    return cfg, dataloader


def grid_and_save(volume, label_list, out_dir, prefix, vis=None):
    # volume: BCDHW
    # label_list: list of tensors [B, C, D, H, W] or [B, 1, D, H, W] where some may be affinity
    # We'll build a canvas: volume (converted to 3 channel), then label(s)

    # Permute: BCDHW -> B D C H W -> for each output (get D slices)
    if isinstance(volume, np.ndarray):
        volume = torch.from_numpy(volume)

    B, C, D, H, W = volume.shape
    # denormalize? The dataset already normalized; if not, this will still show fine
    # Expand to 3 channels
    if C == 1:
        volume_rgb = volume.expand(B, 3, D, H, W)
    else:
        # choose first 3 channels if present
        volume_rgb = volume[:, :3, :, :, :]

    # Convert to grid for each sample and each D slice
    # We will show consecutive slices limited to first 8
    max_slices = min(8, D)

    for i in range(B):
        volume_slices = volume_rgb[i]  # C,D,H,W
        # volume_slices => D,C,H,W
        volume_slices = volume_slices.permute(1, 0, 2, 3)[:max_slices]

        # Build label visuals for each label in label_list
        label_visuals = []
        for lbl in label_list:
            # lbl: B,C,D,H,W or B,D,H,W depending on topt
            if isinstance(lbl, np.ndarray):
                lbl = torch.from_numpy(lbl)
            lbl_i = lbl[i]
            if lbl_i.ndim == 4:  # C,D,H,W
                lbl_i = lbl_i
                # convert to 3-channel heatmap
                if lbl_i.shape[0] == 1:
                    # use semantic color mapping if available from Visualizer
                    if vis is not None:
                        # get_semantic_map expects topt string available in the cfg; we try to use it
                        try:
                            # convert to BCDHW by adding batch dim
                            lbl_tmp = lbl_i.unsqueeze(0)
                            lbl_col = vis.get_semantic_map(lbl_tmp, vis.cfg.MODEL.TARGET_OPT[0], argmax=False)
                            lbl_rgb = lbl_col[0].cpu()
                        except Exception:
                            lbl_rgb = lbl_i.expand(3, lbl_i.shape[1], lbl_i.shape[2], lbl_i.shape[3])
                    else:
                        lbl_rgb = lbl_i.expand(3, lbl_i.shape[1], lbl_i.shape[2], lbl_i.shape[3])
                elif lbl_i.shape[0] >= 3:
                    lbl_rgb = lbl_i[:3]
                else:
                    # duplicate channels
                    lbl_rgb = lbl_i.expand(3, lbl_i.shape[1], lbl_i.shape[2], lbl_i.shape[3])
            elif lbl_i.ndim == 3:  # D,H,W
                lbl_rgb = lbl_i.unsqueeze(0).expand(3, lbl_i.shape[0], lbl_i.shape[1], lbl_i.shape[2])
            else:
                raise ValueError('Unexpected label ndim: ' + str(lbl_i.ndim))
            lbl_slices = lbl_rgb.permute(1, 0, 2, 3)[:max_slices]
            label_visuals.append(lbl_slices)

        canvas = [volume_slices]
        canvas.extend(label_visuals)
        canvas_merge = torch.cat(canvas, 0)
        # normalize and make grid
        grid = vutils.make_grid(canvas_merge, nrow=max_slices, normalize=True, scale_each=True)
        out_path = os.path.join(out_dir, f"{prefix}_sample{i}.png")
        os.makedirs(out_dir, exist_ok=True)
        vutils.save_image(grid, out_path)


def visualize_dataset(dataset_name: str, num_batches: int, out_dir: str):
    cfg, dataloader = build_cfg_and_dataloader(dataset_name)
    vis = Visualizer(cfg)

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for batch_idx, sample in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        # sample.out_input: B,C,D,H,W
        volume = sample.out_input  # torch Tensor
        label_list = sample.out_target_l  # list of torch tensors

        # convert torch to cpu tensors
        volume = volume.cpu()
        label_list = [x.cpu() for x in label_list]

        # Use Visualizer to ensure pre-processing is consistent
        # Visualizer takes volume, label, output, weight, iter_total, writer
        # For output, use label as a placeholder to generate visuals consistently
        # For weight: use sample.out_weight_l
        weights = sample.out_weight_l
        weights = [[w.cpu() for w in weights[k]] for k in range(len(weights))]

        # For each sample in batch, build a grid and save
        # We'll feed the whole batch to a custom save function
        prefix = f"{dataset_name}_batch{batch_idx}"
        try:
            # Note: label_list may contain multiple entries for different targets
            # We will show only the first two (commmonly semantic label and affinity)
            label_vis = [label_list[0]]
            if len(label_list) > 1:
                label_vis.append(label_list[1])
        except Exception as e:
            print('Error preparing labels', e)
            label_vis = label_list

        grid_and_save(volume, label_vis, out_dir, prefix, vis=vis)
        print(f"Saved visualizations for {dataset_name} batch {batch_idx} to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize training samples for AC3-AC4 and EM')
    parser.add_argument('--dataset', type=str, default='em', choices=['em', 'ac3'], help='Dataset to visualize')
    parser.add_argument('--num_batches', type=int, default=2, help='Number of minibatches to visualize')
    parser.add_argument('--out_dir', type=str, default='outputs/visualize', help='Output directory for images')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for dataloader')
    args = parser.parse_args()

    visualize_dataset(args.dataset, args.num_batches, args.out_dir)
