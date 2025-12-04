"""
Script to prepare EM dataset for training.
Extracts volumes from the nested h5 structure and saves them in the format
expected by the VolumeDataset dataloader.
"""
import os
import h5py
import numpy as np
from pathlib import Path


def prepare_em_data(input_h5_path: str, output_dir: str, train_split: float = 0.8):
    """
    Extract EM volumes from nested h5 file and save as separate h5 files.
    
    The input h5 has structure:
        volumes/our1/images: (125, 1250, 1250) uint8
        volumes/our1/labels: (125, 1250, 1250) uint16
        volumes/our2/images: (125, 1250, 1250) uint8
        volumes/our2/labels: (125, 1250, 1250) uint16
    
    Output files:
        EM_train_inputs.h5: training images (our1 full volume)
        EM_train_labels.h5: training labels (our1 full volume)
        EM_val_inputs.h5: validation images (our2 full volume)
        EM_val_labels.h5: validation labels (our2 full volume)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(input_h5_path, 'r') as f:
        # Use our1 for training, our2 for validation
        print("Loading our1 (training)...")
        train_images = np.array(f['volumes']['our1']['images'])
        train_labels = np.array(f['volumes']['our1']['labels'])
        
        print("Loading our2 (validation)...")
        val_images = np.array(f['volumes']['our2']['images'])
        val_labels = np.array(f['volumes']['our2']['labels'])
        
    print(f"Training images shape: {train_images.shape}, dtype: {train_images.dtype}")
    print(f"Training labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")
    print(f"Training labels unique values: {len(np.unique(train_labels))}")
    
    print(f"Validation images shape: {val_images.shape}, dtype: {val_images.dtype}")
    print(f"Validation labels shape: {val_labels.shape}, dtype: {val_labels.dtype}")
    print(f"Validation labels unique values: {len(np.unique(val_labels))}")
    
    # Save training data
    train_input_path = os.path.join(output_dir, 'EM_train_inputs.h5')
    print(f"Saving training inputs to {train_input_path}")
    with h5py.File(train_input_path, 'w') as f:
        f.create_dataset('main', data=train_images, compression='gzip')
    
    train_label_path = os.path.join(output_dir, 'EM_train_labels.h5')
    print(f"Saving training labels to {train_label_path}")
    with h5py.File(train_label_path, 'w') as f:
        f.create_dataset('main', data=train_labels, compression='gzip')
    
    # Save validation data
    val_input_path = os.path.join(output_dir, 'EM_val_inputs.h5')
    print(f"Saving validation inputs to {val_input_path}")
    with h5py.File(val_input_path, 'w') as f:
        f.create_dataset('main', data=val_images, compression='gzip')
    
    val_label_path = os.path.join(output_dir, 'EM_val_labels.h5')
    print(f"Saving validation labels to {val_label_path}")
    with h5py.File(val_label_path, 'w') as f:
        f.create_dataset('main', data=val_labels, compression='gzip')
    
    print("\nData preparation complete!")
    print(f"Files saved to: {output_dir}")
    

def prepare_em_data_combined(input_h5_path: str, output_dir: str, val_slices: int = 25):
    """
    Alternative: Combine both volumes for training with a portion held out for validation.
    
    Args:
        input_h5_path: Path to the em_volumes.h5 file
        output_dir: Output directory for prepared data
        val_slices: Number of slices from each volume to use for validation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(input_h5_path, 'r') as f:
        # Load both volumes
        print("Loading volumes...")
        our1_images = np.array(f['volumes']['our1']['images'])
        our1_labels = np.array(f['volumes']['our1']['labels'])
        our2_images = np.array(f['volumes']['our2']['images'])
        our2_labels = np.array(f['volumes']['our2']['labels'])
    
    # Split: use last val_slices from each volume for validation
    train_images = np.concatenate([
        our1_images[:-val_slices], 
        our2_images[:-val_slices]
    ], axis=0)
    train_labels = np.concatenate([
        our1_labels[:-val_slices], 
        our2_labels[:-val_slices]
    ], axis=0)
    
    val_images = np.concatenate([
        our1_images[-val_slices:], 
        our2_images[-val_slices:]
    ], axis=0)
    val_labels = np.concatenate([
        our1_labels[-val_slices:], 
        our2_labels[-val_slices:]
    ], axis=0)
    
    print(f"Training data: images {train_images.shape}, labels {train_labels.shape}")
    print(f"Validation data: images {val_images.shape}, labels {val_labels.shape}")
    
    # Save files
    with h5py.File(os.path.join(output_dir, 'EM_combined_train_inputs.h5'), 'w') as f:
        f.create_dataset('main', data=train_images, compression='gzip')
    
    with h5py.File(os.path.join(output_dir, 'EM_combined_train_labels.h5'), 'w') as f:
        f.create_dataset('main', data=train_labels, compression='gzip')
    
    with h5py.File(os.path.join(output_dir, 'EM_combined_val_inputs.h5'), 'w') as f:
        f.create_dataset('main', data=val_images, compression='gzip')
    
    with h5py.File(os.path.join(output_dir, 'EM_combined_val_labels.h5'), 'w') as f:
        f.create_dataset('main', data=val_labels, compression='gzip')
    
    print(f"\nCombined data preparation complete!")
    print(f"Files saved to: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare EM dataset for training')
    parser.add_argument('--input', type=str, default='datasets/EM/em_volumes.h5',
                        help='Path to input h5 file')
    parser.add_argument('--output', type=str, default='datasets/EM/',
                        help='Output directory')
    parser.add_argument('--mode', type=str, default='split', choices=['split', 'combined'],
                        help='split: our1 for train, our2 for val; combined: mix both volumes')
    
    args = parser.parse_args()
    
    if args.mode == 'split':
        prepare_em_data(args.input, args.output)
    else:
        prepare_em_data_combined(args.input, args.output)
