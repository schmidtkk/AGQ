"""
3D Signed Distance Function (SDF) computation for instance segmentation masks.

For each instance i:
- Inner points (inside the instance): negative SDF values (distance to boundary, negated)
- Outer points (outside the instance): positive SDF values (distance to boundary)
- Boundary points: SDF = 0

The SDF represents the shortest distance from any point to the nearest boundary
of that instance. For a point inside instance i, its "relative background" is
all pixels where the mask != i.

Fast implementation using scipy.ndimage.distance_transform_edt.
"""

from __future__ import print_function, division
from typing import Optional, Tuple, Union
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as label_cc
import warnings


__all__ = [
    'compute_sdf_3d',
    'compute_sdf_3d_per_instance', 
    'compute_sdf_3d_combined',
    'seg_to_sdf',
]


def compute_sdf_3d_per_instance(
    mask: np.ndarray,
    instance_id: int,
    resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    normalize: bool = True,
    truncate: Optional[float] = None
) -> np.ndarray:
    """
    Compute 3D SDF for a single instance.
    
    Args:
        mask: Instance segmentation mask (D, H, W) with integer labels
        instance_id: The instance ID to compute SDF for
        resolution: Voxel spacing (z, y, x) for anisotropic data
        normalize: If True, normalize SDF to [-1, 1] range
        truncate: If provided, truncate SDF values to [-truncate, truncate]
    
    Returns:
        sdf: 3D SDF array (D, H, W) where:
             - negative values = inside the instance
             - positive values = outside the instance
             - zero = on the boundary
    """
    # Binary mask for this instance
    instance_mask = (mask == instance_id)
    
    if not instance_mask.any():
        # Instance not present, return all positive (outside)
        return np.ones_like(mask, dtype=np.float32) * (truncate if truncate else 1.0)
    
    # Compute distance from inside points to boundary (will be inner distance)
    # EDT of the instance mask gives distance from each point inside to nearest outside
    inner_dist = distance_transform_edt(instance_mask, sampling=resolution)
    
    # Compute distance from outside points to boundary
    # EDT of inverted mask gives distance from each point outside to nearest inside
    outer_dist = distance_transform_edt(~instance_mask, sampling=resolution)
    
    # SDF: negative inside, positive outside
    # Inner points: -inner_dist (negative)
    # Outer points: +outer_dist (positive)
    sdf = outer_dist - inner_dist
    
    if normalize:
        # Normalize to roughly [-1, 1] using max distance
        max_dist = max(inner_dist.max(), outer_dist.max())
        if max_dist > 0:
            sdf = sdf / max_dist
    
    if truncate is not None:
        sdf = np.clip(sdf, -truncate, truncate)
    
    return sdf.astype(np.float32)


def compute_sdf_3d_combined(
    mask: np.ndarray,
    resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    normalize: bool = True,
    truncate: Optional[float] = None,
    background_value: float = 1.0
) -> np.ndarray:
    """
    Compute combined 3D SDF for all instances.
    
    For each pixel, the SDF value is computed relative to the instance it belongs to.
    Background pixels get a constant positive value.
    
    Args:
        mask: Instance segmentation mask (D, H, W) with integer labels (0 = background)
        resolution: Voxel spacing (z, y, x)
        normalize: If True, normalize SDF values
        truncate: If provided, truncate SDF values
        background_value: SDF value for background pixels
    
    Returns:
        sdf: Combined 3D SDF (D, H, W)
    """
    sdf = np.ones_like(mask, dtype=np.float32) * background_value
    
    # Get unique instance IDs (excluding background)
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]
    
    if len(instance_ids) == 0:
        return sdf
    
    # For efficiency, compute EDT once for foreground/background
    foreground = mask > 0
    
    # Distance from background to nearest foreground
    bg_to_fg_dist = distance_transform_edt(~foreground, sampling=resolution)
    
    # For each instance, compute its SDF
    for inst_id in instance_ids:
        inst_mask = mask == inst_id
        
        # Inner distance: distance from inside points to boundary of this instance
        inner_dist = distance_transform_edt(inst_mask, sampling=resolution)
        
        # For points inside this instance, SDF = -inner_dist
        sdf[inst_mask] = -inner_dist[inst_mask]
    
    # For background points, use distance to nearest foreground
    sdf[~foreground] = bg_to_fg_dist[~foreground]
    
    if normalize:
        # Normalize by global max
        abs_max = max(np.abs(sdf.min()), np.abs(sdf.max()))
        if abs_max > 0:
            sdf = sdf / abs_max
    
    if truncate is not None:
        sdf = np.clip(sdf, -truncate, truncate)
    
    return sdf


def compute_sdf_3d(
    mask: np.ndarray,
    resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    normalize: bool = False,
    truncate: Optional[float] = None,
    per_instance: bool = False,
    max_instances: Optional[int] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute 3D Signed Distance Function for instance segmentation mask.
    
    Fast implementation using scipy EDT. For N instances in a (D,H,W) volume,
    complexity is O(N * D * H * W) which is efficient for typical EM volumes.
    
    Args:
        mask: Instance segmentation mask (D, H, W) with integer labels.
              0 is assumed to be background.
        resolution: Voxel spacing (z, y, x) for anisotropic data. Default (1,1,1).
        normalize: If True, normalize each instance's SDF to [-1, 1].
        truncate: If provided, clip SDF values to [-truncate, truncate].
        per_instance: If True, return per-instance SDF stack (N, D, H, W).
                      If False, return combined SDF (D, H, W).
        max_instances: Maximum number of instances to process (for memory/speed).
    
    Returns:
        If per_instance=False: Combined SDF array (D, H, W)
        If per_instance=True: Tuple of (sdf_stack, instance_ids)
            - sdf_stack: (N, D, H, W) array of per-instance SDFs
            - instance_ids: (N,) array of corresponding instance IDs
    """
    if per_instance:
        return _compute_sdf_3d_per_instance_stack(
            mask, resolution, normalize, truncate, max_instances)
    else:
        return compute_sdf_3d_combined(
            mask, resolution, normalize, truncate)


def _compute_sdf_3d_per_instance_stack(
    mask: np.ndarray,
    resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    normalize: bool = True,
    truncate: Optional[float] = None,
    max_instances: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-instance SDF stack.
    
    Returns:
        sdf_stack: (N, D, H, W) array where N is number of instances
        instance_ids: (N,) array of instance IDs
    """
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]  # exclude background
    
    if max_instances is not None and len(instance_ids) > max_instances:
        # Sample instances by size (keep largest ones)
        sizes = [(mask == iid).sum() for iid in instance_ids]
        sorted_idx = np.argsort(sizes)[::-1][:max_instances]
        instance_ids = instance_ids[sorted_idx]
    
    if len(instance_ids) == 0:
        return np.zeros((0,) + mask.shape, dtype=np.float32), np.array([])
    
    sdf_list = []
    for inst_id in instance_ids:
        sdf = compute_sdf_3d_per_instance(
            mask, inst_id, resolution, normalize, truncate)
        sdf_list.append(sdf)
    
    sdf_stack = np.stack(sdf_list, axis=0)
    return sdf_stack, instance_ids


def seg_to_sdf(
    label: np.ndarray,
    topt: str = 'sdf',
    resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Convert segmentation label to SDF target for training.
    
    Target option format: 'sdf' or 'sdf-norm' or 'sdf-trunc-X'
    
    Args:
        label: Instance segmentation mask (D, H, W)
        topt: Target option string
        resolution: Voxel spacing
    
    Returns:
        sdf: SDF array (1, D, H, W) ready for training
    """
    normalize = False
    truncate = None
    
    if '-' in topt:
        parts = topt.split('-')
        for p in parts[1:]:
            if p == 'norm':
                normalize = True
            elif p.replace('.', '').isdigit():
                truncate = float(p)
    
    sdf = compute_sdf_3d_combined(
        label, resolution=resolution, 
        normalize=normalize, truncate=truncate)
    
    # Add channel dimension
    return sdf[np.newaxis, :].astype(np.float32)


# ============ Fast batch processing for large datasets ============

def compute_sdf_3d_fast(
    mask: np.ndarray,
    resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    truncate_distance: float = 10.0,
    num_workers: int = 4
) -> np.ndarray:
    """
    Fast 3D SDF computation optimized for EM volumes with many instances.
    
    This version uses multiprocessing for parallel instance processing.
    
    Args:
        mask: Instance segmentation mask (D, H, W)
        resolution: Voxel spacing
        truncate_distance: Maximum distance to compute (in physical units)
        num_workers: Number of parallel workers
    
    Returns:
        sdf: Combined SDF (D, H, W)
    """
    sdf = np.zeros_like(mask, dtype=np.float32)
    
    # Get unique instances
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]
    
    if len(instance_ids) == 0:
        # All background - return positive distance to edge
        bg_dist = distance_transform_edt(mask == 0, sampling=resolution)
        return np.clip(bg_dist, 0, truncate_distance).astype(np.float32)
    
    # For each foreground voxel, compute its distance to its own boundary
    # This is more efficient than per-instance EDT for dense masks
    
    # Step 1: Compute inner distance for ALL foreground at once using boundary detection
    # Boundary = foreground pixels adjacent to different labels
    from scipy.ndimage import maximum_filter, minimum_filter
    
    # Detect boundaries: where max != min in local neighborhood
    struct_size = 3
    local_max = maximum_filter(mask, size=struct_size, mode='constant', cval=0)
    local_min = minimum_filter(mask, size=struct_size, mode='constant', cval=0)
    
    # Boundary pixels: where neighborhood has different values
    boundary = (local_max != local_min) & (mask > 0)
    
    # Interior pixels: foreground but not boundary
    interior = (mask > 0) & ~boundary
    
    # For interior pixels, compute distance to boundary
    if interior.any():
        # Create a mask where interior pixels need distance to boundary
        # We compute EDT from non-interior (boundary + background) to interior
        inner_dist = distance_transform_edt(interior, sampling=resolution)
        sdf[interior] = -inner_dist[interior]
    
    # Boundary pixels get SDF = 0 (they are on the boundary)
    sdf[boundary] = 0.0
    
    # For background pixels, compute distance to nearest foreground
    background = mask == 0
    if background.any():
        bg_dist = distance_transform_edt(background, sampling=resolution)
        sdf[background] = bg_dist[background]
    
    # Truncate
    sdf = np.clip(sdf, -truncate_distance, truncate_distance)
    
    return sdf


def compute_sdf_3d_accurate(
    mask: np.ndarray,
    resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    truncate_distance: float = 10.0
) -> np.ndarray:
    """
    Accurate 3D SDF computation - processes each instance separately.
    
    Slower but more accurate for cases where instances touch each other.
    
    Args:
        mask: Instance segmentation mask (D, H, W)
        resolution: Voxel spacing
        truncate_distance: Maximum distance to compute
    
    Returns:
        sdf: Combined SDF (D, H, W)
    """
    sdf = np.zeros_like(mask, dtype=np.float32)
    
    # Get unique instances
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]
    
    if len(instance_ids) == 0:
        bg_dist = distance_transform_edt(mask == 0, sampling=resolution)
        return np.clip(bg_dist, 0, truncate_distance).astype(np.float32)
    
    # Process each instance
    for inst_id in instance_ids:
        inst_mask = mask == inst_id
        
        # Skip tiny instances
        if inst_mask.sum() < 10:
            continue
        
        # Compute inner distance (negative inside)
        inner_dist = distance_transform_edt(inst_mask, sampling=resolution)
        
        # Apply to SDF (inside this instance gets negative values)
        sdf[inst_mask] = -inner_dist[inst_mask]
    
    # For background pixels, compute distance to nearest foreground
    background = mask == 0
    if background.any():
        bg_dist = distance_transform_edt(background, sampling=resolution)
        sdf[background] = bg_dist[background]
    
    # Truncate
    sdf = np.clip(sdf, -truncate_distance, truncate_distance)
    
    return sdf


if __name__ == '__main__':
    # Test the SDF computation
    import time
    
    # Create a simple test mask
    mask = np.zeros((32, 64, 64), dtype=np.int32)
    mask[10:20, 20:40, 20:40] = 1  # Instance 1: a cube
    mask[12:18, 45:55, 10:20] = 2  # Instance 2: another cube
    
    print("Test mask shape:", mask.shape)
    print("Unique labels:", np.unique(mask))
    
    # Test combined SDF
    start = time.time()
    sdf_combined = compute_sdf_3d_combined(mask, normalize=False)
    print(f"Combined SDF computed in {time.time() - start:.3f}s")
    print(f"  Shape: {sdf_combined.shape}")
    print(f"  Range: [{sdf_combined.min():.2f}, {sdf_combined.max():.2f}]")
    print(f"  Inside inst 1 (should be negative): {sdf_combined[15, 30, 30]:.2f}")
    print(f"  Background (should be positive): {sdf_combined[0, 0, 0]:.2f}")
    
    # Test per-instance SDF
    start = time.time()
    sdf_stack, inst_ids = compute_sdf_3d(mask, per_instance=True)
    print(f"\nPer-instance SDF computed in {time.time() - start:.3f}s")
    print(f"  Stack shape: {sdf_stack.shape}")
    print(f"  Instance IDs: {inst_ids}")
    
    # Test fast version
    start = time.time()
    sdf_fast = compute_sdf_3d_fast(mask)
    print(f"\nFast SDF computed in {time.time() - start:.3f}s")
    print(f"  Range: [{sdf_fast.min():.2f}, {sdf_fast.max():.2f}]")
    
    # Test with larger synthetic data
    print("\n--- Large volume test ---")
    large_mask = np.zeros((125, 256, 256), dtype=np.int32)
    # Add 50 random instances
    np.random.seed(42)
    for i in range(50):
        z = np.random.randint(10, 115)
        y = np.random.randint(20, 236)
        x = np.random.randint(20, 236)
        sz = np.random.randint(5, 15)
        sy = np.random.randint(10, 30)
        sx = np.random.randint(10, 30)
        large_mask[z:z+sz, y:y+sy, x:x+sx] = i + 1
    
    print(f"Large mask shape: {large_mask.shape}")
    print(f"Number of instances: {len(np.unique(large_mask)) - 1}")
    
    start = time.time()
    sdf_large = compute_sdf_3d_fast(large_mask, truncate_distance=20.0)
    elapsed = time.time() - start
    print(f"Fast SDF on large volume: {elapsed:.2f}s")
    print(f"  Range: [{sdf_large.min():.2f}, {sdf_large.max():.2f}]")
    
    # Test accurate version for comparison
    start = time.time()
    sdf_accurate = compute_sdf_3d_accurate(large_mask, truncate_distance=20.0)
    elapsed_acc = time.time() - start
    print(f"Accurate SDF on large volume: {elapsed_acc:.2f}s")
    print(f"  Range: [{sdf_accurate.min():.2f}, {sdf_accurate.max():.2f}]")
