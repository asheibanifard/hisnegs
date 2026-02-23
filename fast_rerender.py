#!/usr/bin/env python3
"""Fast GPU renderer for all ablation volumes at original GT resolution (100x647x813).
Uses large GPU chunks on the RTX 8000 (48GB VRAM) to maximize throughput."""

import sys, torch, time
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

device = 'cuda'
BASE = Path('/workspace/hisnegs/src/ablation_results/loss_components')
ABLATIONS = ['baseline','no_grad_loss','no_tube_reg','no_cross_reg','no_scale_reg','no_regularizers','reconstruction_only']
Z_RES, Y_RES, X_RES = 100, 647, 813  # original GT dimensions

def render_one(checkpoint_path, output_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    means = ckpt['means'].to(device)
    log_scales = ckpt['log_scales'].to(device)
    quaternions = ckpt['quaternions'].to(device)
    if 'log_amplitudes' in ckpt:
        log_amplitudes = ckpt['log_amplitudes'].to(device)
    else:
        log_amplitudes = ckpt['log_intensities'].to(device)
    
    amplitudes = torch.exp(log_amplitudes.clamp(-10.0, 6.0))
    scales = torch.exp(log_scales.clamp(-10.0, 2.0))
    
    # Normalize quaternions and build rotation matrices
    quaternions = quaternions / (quaternions.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    K = means.shape[0]
    
    R = torch.zeros(K, 3, 3, device=device)
    R[:, 0, 0] = 1 - 2*(y*y + z*z);  R[:, 0, 1] = 2*(x*y - w*z);  R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z);      R[:, 1, 1] = 1 - 2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y);      R[:, 2, 1] = 2*(y*z + w*x);  R[:, 2, 2] = 1 - 2*(x*x + y*y)
    
    S = torch.diag_embed(scales)
    covariances = R @ S @ S @ R.transpose(-2, -1) + 1e-6 * torch.eye(3, device=device).unsqueeze(0)
    L_chol = torch.linalg.cholesky(covariances)
    
    # Create grid
    x_ax = torch.linspace(-1, 1, X_RES, device=device)
    y_ax = torch.linspace(-1, 1, Y_RES, device=device)
    z_ax = torch.linspace(-1, 1, Z_RES, device=device)
    zz, yy, xx = torch.meshgrid(z_ax, y_ax, x_ax, indexing='ij')
    grid_points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    
    total = grid_points.shape[0]
    chunk_size = 65536  # 16x larger chunks for RTX 8000
    volume_flat = torch.zeros(total, device=device)
    
    with torch.no_grad():
        for i in tqdm(range(0, total, chunk_size), desc="  GPU eval"):
            end = min(i + chunk_size, total)
            pts = grid_points[i:end]
            diff = pts.unsqueeze(1) - means.unsqueeze(0)  # (N, K, 3)
            y_val = torch.linalg.solve_triangular(L_chol, diff.unsqueeze(-1), upper=False)
            mahal = (y_val.squeeze(-1) ** 2).sum(dim=-1)
            vals = (torch.exp(-0.5 * mahal) * amplitudes.unsqueeze(0)).sum(dim=1)
            volume_flat[i:end] = vals
    
    volume = volume_flat.reshape(Z_RES, Y_RES, X_RES).cpu()
    torch.save(volume, output_path)
    
    del volume_flat, grid_points, L_chol, covariances
    torch.cuda.empty_cache()
    return volume.shape

print(f"Rendering all ablations at {Z_RES}×{Y_RES}×{X_RES} on {device}")
print(f"Total voxels per volume: {Z_RES*Y_RES*X_RES:,}")
print()

t0_all = time.time()
for abl in ABLATIONS:
    ckpt = BASE / abl / 'checkpoints' / 'model_step20000.pt'
    out = BASE / abl / 'rendered_volume.pt'
    print(f"=== {abl} ===")
    t0 = time.time()
    shape = render_one(str(ckpt), str(out))
    dt = time.time() - t0
    print(f"  Done: {list(shape)} in {dt:.0f}s\n")

print(f"\n✓ All done in {time.time()-t0_all:.0f}s")
