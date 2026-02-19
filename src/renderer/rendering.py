#!/usr/bin/env python3
"""
NeuroSGM — Neurite Sparse Gaussian Mixture: MIP Rendering Pipeline
====================================================================

Renders 3D Gaussians G(μ, Σ, intensity) as Maximum Intensity Projections.

Formulation
-----------
Each Gaussian is parameterised by:
    μ          : (3,)    centre in world space
    Σ          : (3, 3)  anisotropic covariance (per-axis scales + rotation)
    intensity  : scalar  signal amplitude in [0, 1]   ← NO opacity

Splatting pipeline for one projection view k:

    1) Transform to camera frame:
           μ_cam = R·μ + T
           Σ_cam = R·Σ·Rᵀ

    2) Project to 2D image plane:
           μ₂D = (fx·μx/μz + cx,  fy·μy/μz + cy)
           J   = [[fx/z,  0,  -fx·x/z²],
                  [0,  fy/z,  -fy·y/z²]]
           Σ₂D = J·Σ_cam·Jᵀ

    3) Evaluate 2D Gaussian at every pixel p = (u, v):
           G₂D(p; μ₂D, Σ₂D) = exp(-½ (p-μ₂D)ᵀ Σ₂D⁻¹ (p-μ₂D))

    4) MIP splatting — replaces alpha compositing entirely:
           I_k(u,v) = max_i [ intensity_i · G₂D_i(u,v) ]

       Loss over all M projection views (matches formulation):
           L = Σ_{k=1}^{M} ‖ I_k(u,v) − max_i [ intensity_i · G₂D_i(u,v) ] ‖²

Design decisions
----------------
- No opacity parameter. Intensity is the sole scalar per Gaussian.
- MIP naturally models fluorescence microscopy: brightest structure wins.
- Training iterates over ALL M projection views per epoch — directly matches
  the Σ_k outer sum in the formulation (not random pixel sampling).
- log_scales is (K, 3): per-axis anisotropic scales essential for neurites.
- Differentiable MIP via soft-max approximation (β controls sharpness).
"""

from __future__ import annotations

import math
import os
import csv
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .splat_cuda_wrapper import CUDASplattingTrainer
except ImportError:
    try:
        from splat_cuda_wrapper import CUDASplattingTrainer
    except ImportError:
        CUDASplattingTrainer = None


# ===================================================================
#  Camera (pinhole)
# ===================================================================
@dataclass
class Camera:
    """
    Pinhole camera intrinsics.

    Attributes
    ----------
    fx, fy        : focal lengths in pixels
    cx, cy        : principal point in pixels
    width, height : image resolution
    near, far     : depth clipping planes
    """
    fx:     float
    fy:     float
    cx:     float
    cy:     float
    width:  int
    height: int
    near:   float = 0.1
    far:    float = 100.0

    @property
    def K(self) -> torch.Tensor:
        """3×3 intrinsic matrix."""
        return torch.tensor([
            [self.fx,    0.0, self.cx],
            [   0.0, self.fy, self.cy],
            [   0.0,    0.0,    1.0 ],
        ], dtype=torch.float32)

    @classmethod
    def from_fov(
        cls,
        fov_x_deg: float,
        width:     int,
        height:    int,
        near:      float = 0.1,
        far:       float = 100.0,
    ) -> "Camera":
        """Construct from horizontal field-of-view (degrees)."""
        fx = width / (2.0 * math.tan(math.radians(fov_x_deg) / 2.0))
        return cls(
            fx=fx, fy=fx,
            cx=width  / 2.0,
            cy=height / 2.0,
            width=width, height=height,
            near=near, far=far,
        )


# ===================================================================
#  Gaussian parameter container  (intensity-only, no opacity)
# ===================================================================
@dataclass
class GaussianParameters:
    """
    Intensity-only 3D Gaussian primitives — no opacity field.

    Attributes
    ----------
    means       : (K, 3)    world-space centres μ_k
    covariances : (K, 3, 3) anisotropic covariance matrices Σ_k
    intensities : (K,)      signal amplitude in [0, 1]
    """
    means:       torch.Tensor   # (K, 3)
    covariances: torch.Tensor   # (K, 3, 3)
    intensities: torch.Tensor   # (K,)


# ===================================================================
#  Step 1 — Transform to camera frame
#
#      μ_cam = R·μ + T    →  batched: means @ Rᵀ + T      (K, 3)
#      Σ_cam = R·Σ·Rᵀ    →  batched: R @ Σ @ Rᵀ           (K, 3, 3)
# ===================================================================
def transform_to_camera(
    means:       torch.Tensor,
    covariances: torch.Tensor,
    R:           torch.Tensor,
    T:           torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform Gaussians from world to camera coordinates.

    Parameters
    ----------
    means       : (K, 3)
    covariances : (K, 3, 3)
    R           : (3, 3)  world-to-camera rotation
    T           : (3,)    world-to-camera translation

    Returns
    -------
    means_cam : (K, 3)
    cov_cam   : (K, 3, 3)
    """
    means_cam = means @ R.T + T.unsqueeze(0)                        # (K, 3)
    cov_cam   = R.unsqueeze(0) @ covariances @ R.T.unsqueeze(0)     # (K, 3, 3)
    return means_cam, cov_cam


# ===================================================================
#  Step 2 — Project to 2D
#
#      μ₂D = (fx·x/z + cx,  fy·y/z + cy)
#
#      J   = [[fx/z,  0,  -fx·x/z²],
#             [0,  fy/z,  -fy·y/z²]]
#
#      Σ₂D = J · Σ_cam · Jᵀ
# ===================================================================
def compute_projection_jacobian(
    means_cam: torch.Tensor,
    fx:        float,
    fy:        float,
) -> torch.Tensor:
    """
    Per-Gaussian Jacobian J = ∂(u,v)/∂(x,y,z).

    Parameters
    ----------
    means_cam : (K, 3)
    fx, fy    : focal lengths

    Returns
    -------
    J : (K, 2, 3)
    """
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    z_s  = z.clamp(min=1e-6)
    z2_s = (z * z).clamp(min=1e-12)

    K = x.shape[0]
    J = torch.zeros(K, 2, 3, device=means_cam.device, dtype=means_cam.dtype)
    J[:, 0, 0] =  fx / z_s        # ∂u/∂x =  fx/z
    J[:, 0, 2] = -fx * x / z2_s   # ∂u/∂z = -fx·x/z²
    J[:, 1, 1] =  fy / z_s        # ∂v/∂y =  fy/z
    J[:, 1, 2] = -fy * y / z2_s   # ∂v/∂z = -fy·y/z²
    return J  # (K, 2, 3)


def project_to_2d(
    means_cam: torch.Tensor,
    cov_cam:   torch.Tensor,
    camera:    Camera,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project camera-frame Gaussians to the image plane.

    Returns
    -------
    means_2d : (K, 2)    pixel-space centres
    cov_2d   : (K, 2, 2) Σ₂D = J Σ_cam Jᵀ
    depths   : (K,)      z depths
    """
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    z_s = z.clamp(min=1e-6)

    u = camera.fx * x / z_s + camera.cx
    v = camera.fy * y / z_s + camera.cy
    means_2d = torch.stack([u, v], dim=-1)                          # (K, 2)

    J      = compute_projection_jacobian(means_cam, camera.fx, camera.fy)
    cov_2d = J @ cov_cam @ J.transpose(-2, -1)                      # (K, 2, 2)

    # Numerical stability: small isotropic floor
    eye    = torch.eye(2, device=cov_2d.device, dtype=cov_2d.dtype).unsqueeze(0)
    cov_2d = cov_2d + 1e-4 * eye

    return means_2d, cov_2d, z


# ===================================================================
#  Step 3 — Evaluate 2D Gaussian
#
#      G₂D(p; μ₂D, Σ₂D) = exp(-½ (p-μ₂D)ᵀ Σ₂D⁻¹ (p-μ₂D))
# ===================================================================
def evaluate_gaussian_2d(
    pixels:   torch.Tensor,
    means_2d: torch.Tensor,
    cov_2d:   torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate 2D Gaussians at pixel locations.

    Parameters
    ----------
    pixels   : (N, 2)    query pixel coordinates (u, v)
    means_2d : (K, 2)
    cov_2d   : (K, 2, 2)

    Returns
    -------
    values : (N, K)   G₂D ∈ (0, 1]
    """
    # (N, K, 2) displacement vectors  p - μ
    diff = pixels[:, None, :] - means_2d[None, :, :]

    # Analytic 2×2 matrix inverse
    a, b    = cov_2d[:, 0, 0], cov_2d[:, 0, 1]
    c, d    = cov_2d[:, 1, 0], cov_2d[:, 1, 1]
    inv_det = 1.0 / (a * d - b * c).clamp(min=1e-12)

    cov_inv = torch.stack([
        torch.stack([ d * inv_det, -b * inv_det], dim=-1),
        torch.stack([-c * inv_det,  a * inv_det], dim=-1),
    ], dim=-2)  # (K, 2, 2)

    # Mahalanobis distance (p-μ)ᵀ Σ⁻¹ (p-μ)  →  (N, K)
    tmp   = torch.einsum('nki,kij->nkj', diff, cov_inv)
    mahal = (tmp * diff).sum(dim=-1)

    return torch.exp(-0.5 * mahal)  # (N, K)


# ===================================================================
#  Step 4 — MIP Splatting  (no opacity)
#
#  Hard MIP:  I(u,v) = max_i [ intensity_i · G₂D_i(u,v) ]
#
#  Differentiable soft-max approximation:
#      I(u,v) = Σᵢ softmax(β · aᵢ · G₂D_i) · aᵢ · G₂D_i
#
#  where aᵢ = intensity_i ∈ [0,1] and β is the sharpness.
#  As β → ∞ this recovers the hard max exactly.
# ===================================================================
def splat_mip(
    pixels:      torch.Tensor,
    means_2d:    torch.Tensor,
    cov_2d:      torch.Tensor,
    intensities: torch.Tensor,
    beta:        float = 10.0,
    chunk_size:  int   = 2048,
) -> torch.Tensor:
    """
    Differentiable MIP splatting over all Gaussians.

    Memory layout
    -------------
    Peak per chunk: 3 × (n × K) floats.
    With chunk_size=4096, K=5000: ~240 MB per chunk.
    Reduce chunk_size if GPU memory is tight.

    Optimisations vs naive version
    --------------------------------
    - cov_inv precomputed once outside loop (saves K × 4 muls per chunk)
    - in-place mul_ folds intensity weighting into G₂D buffer
    - explicit del frees (n,K) tensors immediately after use
    - pixel grid built lazily (no (H*W, 2) tensor held during splatting)

    Parameters
    ----------
    pixels      : (N, 2)   pixel coordinates
    means_2d    : (K, 2)
    cov_2d      : (K, 2, 2)
    intensities : (K,)     signal amplitudes in [0, 1]  — no opacity
    beta        : softmax temperature (higher → sharper MIP)
    chunk_size  : pixels per chunk (decrease to save memory)

    Returns
    -------
    image : (N,)  rendered intensity in [0, 1]
    """
    N   = pixels.shape[0]
    out = torch.zeros(N, device=pixels.device, dtype=pixels.dtype)

    # Pre-compute 2×2 inverse covariance once — reused across all pixel chunks
    a, b    = cov_2d[:, 0, 0], cov_2d[:, 0, 1]
    c, d_v  = cov_2d[:, 1, 0], cov_2d[:, 1, 1]
    inv_det = 1.0 / (a * d_v - b * c).clamp(min=1e-12)
    cov_inv = torch.stack([
        torch.stack([ d_v * inv_det, -b   * inv_det], dim=-1),
        torch.stack([-c   * inv_det,  a   * inv_det], dim=-1),
    ], dim=-2)  # (K, 2, 2)

    for i in range(0, N, chunk_size):
        pix  = pixels[i : i + chunk_size]                 # (n, 2)

        # Displacement (n, K, 2) — only tensor of this size in scope at once
        diff  = pix[:, None, :] - means_2d[None, :, :]
        tmp   = torch.einsum('nki,kij->nkj', diff, cov_inv)
        mahal = (tmp * diff).sum(-1)                       # (n, K)
        del diff, tmp

        # G₂D · intensity in-place — avoids a second (n,K) allocation
        gauss = torch.exp_(-0.5 * mahal)                  # (n, K)  in-place exp
        del mahal
        gauss.mul_(intensities[None, :])                   # (n, K)  in-place ×intensity

        # Soft-max MIP
        sm  = torch.softmax(beta * gauss, dim=-1)          # (n, K)
        out[i : i + chunk_size] = (sm * gauss).sum(-1)    # (n,)
        del gauss, sm

    return out  # (N,)


def splat_mip_grid(
    H:           int,
    W:           int,
    means_2d:    torch.Tensor,
    cov_2d:      torch.Tensor,
    intensities: torch.Tensor,
    beta:        float = 10.0,
    chunk_size:  int   = 4096,
    device:      torch.device = None,
) -> torch.Tensor:
    """
    MIP splat directly over a pixel grid without materialising the full
    (H*W, 2) pixel coordinate tensor.

    Generates pixel coordinates row-batch by row-batch, so peak extra
    memory is chunk_size × 2 floats instead of H*W × 2 floats.
    """
    if device is None:
        device = means_2d.device

    N   = H * W
    out = torch.zeros(N, device=device, dtype=means_2d.dtype)

    # Pre-compute cov_inv once
    a, b    = cov_2d[:, 0, 0], cov_2d[:, 0, 1]
    c, d_v  = cov_2d[:, 1, 0], cov_2d[:, 1, 1]
    inv_det = 1.0 / (a * d_v - b * c).clamp(min=1e-12)
    cov_inv = torch.stack([
        torch.stack([ d_v * inv_det, -b   * inv_det], dim=-1),
        torch.stack([-c   * inv_det,  a   * inv_det], dim=-1),
    ], dim=-2)  # (K, 2, 2)

    xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5   # (W,)

    for row_start in range(0, H, chunk_size // max(W, 1)):
        row_end  = min(row_start + chunk_size // max(W, 1), H)
        ys       = torch.arange(row_start, row_end,
                                device=device, dtype=torch.float32) + 0.5  # (r,)
        gy, gx   = torch.meshgrid(ys, xs, indexing='ij')
        pix      = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)   # (r*W, 2)

        i = row_start * W
        j = row_end   * W

        diff  = pix[:, None, :] - means_2d[None, :, :]         # (r*W, K, 2)
        tmp   = torch.einsum('nki,kij->nkj', diff, cov_inv)
        mahal = (tmp * diff).sum(-1)                             # (r*W, K)
        del diff, tmp, pix

        gauss = torch.exp_(-0.5 * mahal)
        del mahal
        gauss.mul_(intensities[None, :])

        sm      = torch.softmax(beta * gauss, dim=-1)
        out[i:j] = (sm * gauss).sum(-1)
        del gauss, sm

    return out  # (H*W,)


# ===================================================================
#  Full MIP projection renderer
# ===================================================================
def render_mip_projection(
    gaussians:  GaussianParameters,
    camera:     Camera,
    R:          torch.Tensor,
    T:          torch.Tensor,
    beta:       float = 10.0,
    chunk_size: int   = 4096,
) -> Tuple[torch.Tensor, int]:
    """
    Render a full MIP projection image from 3D Gaussians.

    Steps: world → camera → 2D project → frustum cull → MIP splat

    Parameters
    ----------
    gaussians  : GaussianParameters  (intensity-only)
    camera     : Camera
    R          : (3, 3)  world-to-camera rotation
    T          : (3,)    world-to-camera translation
    beta       : soft-MIP sharpness
    chunk_size : pixels per chunk

    Returns
    -------
    image     : (H, W)  MIP projection in [0, 1]
    n_visible : int     Gaussians that survived culling
    """
    device = gaussians.means.device
    H, W   = camera.height, camera.width

    # Step 1: Transform
    means_cam, cov_cam = transform_to_camera(
        gaussians.means, gaussians.covariances, R, T)

    # Depth culling
    z       = means_cam[:, 2]
    vis     = (z > camera.near) & (z < camera.far)

    if vis.sum() == 0:
        return torch.zeros(H, W, device=device), 0

    means_cam_v = means_cam[vis]
    cov_cam_v   = cov_cam[vis]
    intens_v    = gaussians.intensities[vis]

    # Step 2: Project
    means_2d, cov_2d, _ = project_to_2d(means_cam_v, cov_cam_v, camera)

    # Frustum culling: 2D footprint must overlap image
    with torch.no_grad():
        a_c, b_c, d_c = cov_2d[:, 0, 0], cov_2d[:, 0, 1], cov_2d[:, 1, 1]
        tr     = a_c + d_c
        disc   = (tr * tr - 4.0 * (a_c * d_c - b_c * b_c)).clamp(min=0.0)
        radius = 3.0 * torch.sqrt(0.5 * (tr + torch.sqrt(disc)).clamp(min=1e-8))
        u_2d, v_2d = means_2d[:, 0], means_2d[:, 1]
        in_img = ((u_2d + radius > 0) & (u_2d - radius < W) &
                  (v_2d + radius > 0) & (v_2d - radius < H))

    means_2d  = means_2d[in_img]
    cov_2d    = cov_2d[in_img]
    intens_v  = intens_v[in_img]
    n_visible = int(in_img.sum().item())

    if n_visible == 0:
        return torch.zeros(H, W, device=device), 0

    # Step 3 & 4: Pixel grid built lazily inside splat — no (H*W,2) tensor held
    image = splat_mip_grid(H, W, means_2d, cov_2d, intens_v,
                           beta=beta, chunk_size=chunk_size,
                           device=device)

    return image.reshape(H, W), n_visible


# ===================================================================
#  Loss functions
# ===================================================================
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L2 loss — matches ‖·‖² in the splatting formulation."""
    return F.mse_loss(pred, target)


def ssim_loss_fn(
    pred:        torch.Tensor,
    target:      torch.Tensor,
    window_size: int   = 11,
    C1:          float = 1e-4,
    C2:          float = 9e-4,
) -> torch.Tensor:
    """
    1 − SSIM on full (H, W) projection images.
    NOTE: requires complete projection images — spatially meaningless
    on pixel subsets.
    """
    p   = pred.unsqueeze(0).unsqueeze(0)    # (1, 1, H, W)
    g   = target.unsqueeze(0).unsqueeze(0)

    coords = torch.arange(window_size, device=pred.device, dtype=torch.float32)
    coords -= window_size // 2
    win     = torch.exp(-coords ** 2 / (2.0 * 1.5 ** 2))
    win     = win.unsqueeze(1) * win.unsqueeze(0)
    win     = (win / win.sum()).unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)
    pad     = window_size // 2

    mu_p  = F.conv2d(p,     win, padding=pad)
    mu_g  = F.conv2d(g,     win, padding=pad)
    sig_p = F.conv2d(p * p, win, padding=pad) - mu_p ** 2
    sig_g = F.conv2d(g * g, win, padding=pad) - mu_g ** 2
    sig_x = F.conv2d(p * g, win, padding=pad) - mu_p * mu_g

    ssim_map = ((2 * mu_p * mu_g + C1) * (2 * sig_x + C2)) / \
               ((mu_p**2 + mu_g**2 + C1) * (sig_p + sig_g + C2))
    return 1.0 - ssim_map.mean()


def psnr_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    return float(-10.0 * math.log10(max(mse, 1e-12)))


def lpips_metric(pred: torch.Tensor, target: torch.Tensor, lpips_model) -> float:
    if lpips_model is None:
        return float("nan")
    with torch.no_grad():
        p = pred.detach().float().clamp(0, 1)[None, None].repeat(1, 3, 1, 1) * 2 - 1
        t = target.detach().float().clamp(0, 1)[None, None].repeat(1, 3, 1, 1) * 2 - 1
        return float(lpips_model(p, t).mean().item())


# ===================================================================
#  Aspect-ratio helpers
# ===================================================================
def compute_aspect_scales(vol_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Scale factors: isotropic [-1,1]³ training space → proportional world space.
    The largest volume dimension maps to [-1, 1].

    Parameters
    ----------
    vol_shape : (Z, Y, X)

    Returns
    -------
    scales : (3,)  [sx, sy, sz]
    """
    Z, Y, X = vol_shape
    m = float(max(X, Y, Z))
    return torch.tensor([X / m, Y / m, Z / m], dtype=torch.float32)


def apply_aspect_correction(
    gaussians:     GaussianParameters,
    aspect_scales: torch.Tensor,
) -> GaussianParameters:
    """Scale Gaussians from isotropic training space to aspect-correct world space."""
    s = aspect_scales.to(gaussians.means.device)
    S = torch.diag(s)
    return GaussianParameters(
        means       = gaussians.means * s.unsqueeze(0),
        covariances = S.unsqueeze(0) @ gaussians.covariances @ S.unsqueeze(0).transpose(-2, -1),
        intensities = gaussians.intensities,
    )


# ===================================================================
#  GT MIP dataset generation from raw 3D volume
# ===================================================================
def load_volume(tif_path: str) -> np.ndarray:
    """Load 3D TIFF and normalise to [0, 1] float32."""
    import tifffile
    vol = tifffile.imread(tif_path).astype(np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(vol, dtype=np.float32)
    return (vol - vmin) / (vmax - vmin)


def _sample_volume_trilinear(vol: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Trilinear sampling at normalised [-1,1]³ coordinates.

    Parameters
    ----------
    vol    : (1, 1, Z, Y, X)
    points : (N, 3)

    Returns
    -------
    values : (N,)
    """
    if vol.ndim == 3:
        vol = vol.unsqueeze(0).unsqueeze(0)
    N    = points.shape[0]
    grid = points.reshape(1, 1, 1, N, 3)
    return F.grid_sample(vol, grid, mode='bilinear',
                         padding_mode='zeros',
                         align_corners=True).reshape(N)


def render_gt_mip(
    vol:           torch.Tensor,
    camera:        Camera,
    R:             torch.Tensor,
    T:             torch.Tensor,
    n_samples:     int                     = 256,
    near:          float                   = 0.5,
    far:           float                   = 6.0,
    aspect_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Render ground-truth MIP from the raw 3D volume via ray marching.

    True MIP: samples n_samples points along each ray and takes the max
    intensity. No opacity, no compositing.

    Parameters
    ----------
    vol           : (Z, Y, X)  normalised volume on device
    camera        : Camera
    R             : (3, 3)  world-to-camera rotation
    T             : (3,)    world-to-camera translation
    n_samples     : samples per ray
    near, far     : ray march depth bounds
    aspect_scales : (3,) undo aspect correction when querying isotropic volume

    Returns
    -------
    mip : (H, W)  in [0, 1]
    """
    device = vol.device
    if vol.ndim == 3:
        vol = vol.unsqueeze(0).unsqueeze(0)

    H, W = camera.height, camera.width
    R_cw = R.T                  # camera-to-world rotation
    T_cw = -R_cw @ T            # camera position in world

    ys = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')

    # Ray directions in world space
    dirs_cam = torch.stack([
        (gx - camera.cx) / camera.fx,
        (gy - camera.cy) / camera.fy,
        torch.ones_like(gx),
    ], dim=-1)  # (H, W, 3)

    dirs_world = (R_cw @ dirs_cam.reshape(-1, 3).T).T   # (H*W, 3)
    dirs_world = dirs_world / (dirs_world.norm(dim=-1, keepdim=True) + 1e-8)

    origin = T_cw.unsqueeze(0)  # (1, 3)
    t_vals = torch.linspace(near, far, n_samples, device=device)

    # Sample points: (H*W, n_samples, 3)
    points = (origin.unsqueeze(1)
              + t_vals[None, :, None] * dirs_world.unsqueeze(1))

    # MIP via chunked ray marching — max along each ray
    N_rays   = H * W
    mip_flat = torch.zeros(N_rays, device=device)
    chunk    = 1024

    for i in range(0, N_rays, chunk):
        j   = min(i + chunk, N_rays)
        pts = points[i:j].reshape(-1, 3)         # (chunk*S, 3)
        if aspect_scales is not None:
            inv_s = 1.0 / aspect_scales.to(pts.device)
            pts   = pts * inv_s.unsqueeze(0)     # back to isotropic [-1,1]³
        vals = _sample_volume_trilinear(vol, pts).reshape(j - i, n_samples)
        mip_flat[i:j] = vals.max(dim=1)[0]       # true MIP: hard max

    return mip_flat.reshape(H, W)


def _orbit_pose(
    elevation_deg: float,
    azimuth_deg:   float,
    radius:        float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Orbit camera extrinsics (R, T) in OpenCV convention."""
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    cam_pos = torch.tensor([
        radius * math.cos(el) * math.sin(az),
        radius * math.sin(el),
        radius * math.cos(el) * math.cos(az),
    ], dtype=torch.float32)

    forward  = -cam_pos / (cam_pos.norm() + 1e-8)
    world_up = torch.tensor([0.0, 1.0, 0.0])
    right    = torch.linalg.cross(forward, world_up)
    if right.norm() < 1e-6:
        world_up = torch.tensor([0.0, 0.0, 1.0])
        right    = torch.linalg.cross(forward, world_up)
    right = right / (right.norm() + 1e-8)
    up    = torch.linalg.cross(right, forward)
    up    = up    / (up.norm()    + 1e-8)

    R = torch.stack([right, -up, forward], dim=0)  # (3,3) OpenCV: x=right, y=-up, z=fwd
    T = -R @ cam_pos                                # (3,)
    return R, T


def generate_camera_poses(
    n_azimuth:            int                   = 12,
    n_elevation:          int                   = 5,
    elevation_range:      Tuple[float, float]   = (-60.0, 60.0),
    radius:               float                 = 3.5,
    include_axis_aligned: bool                  = True,
) -> List[dict]:
    """Generate M orbit camera poses. Returns list of {R, T, elevation, azimuth}."""
    poses = []
    for el in np.linspace(elevation_range[0], elevation_range[1], n_elevation):
        for az in np.linspace(0, 360, n_azimuth, endpoint=False):
            R, T = _orbit_pose(float(el), float(az), radius)
            poses.append({'R': R, 'T': T,
                          'elevation': float(el), 'azimuth': float(az)})
    if include_axis_aligned:
        for el, az in [(0,0),(0,180),(0,90),(0,-90),(89,0),(-89,0)]:
            R, T = _orbit_pose(float(el), float(az), radius)
            poses.append({'R': R, 'T': T,
                          'elevation': float(el), 'azimuth': float(az)})
    return poses


def generate_mip_dataset(
    vol:           torch.Tensor,
    camera:        Camera,
    poses:         List[dict],
    n_ray_samples: int                     = 256,
    near:          float                   = 0.5,
    far:           float                   = 6.0,
    aspect_scales: Optional[torch.Tensor] = None,
) -> List[dict]:
    """
    Render GT MIP projections for all M camera poses.

    Returns list of {image (H,W), R, T, elevation, azimuth}.
    """
    dataset = []
    device  = vol.device
    for idx, pose in enumerate(poses):
        R, T = pose['R'].to(device), pose['T'].to(device)
        mip  = render_gt_mip(vol, camera, R, T,
                             n_samples=n_ray_samples,
                             near=near, far=far,
                             aspect_scales=aspect_scales)
        dataset.append({
            'image':     mip,
            'R':         R,
            'T':         T,
            'elevation': pose['elevation'],
            'azimuth':   pose['azimuth'],
        })
        if (idx + 1) % 10 == 0 or idx == len(poses) - 1:
            print(f"  GT MIP: {idx+1}/{len(poses)} projections rendered")
    return dataset


# ===================================================================
#  MIPSplattingTrainer
#  ─────────────────────────────────────────────────────────────────
#  Learnable parameters:
#      means           (K, 3)   Gaussian centres
#      log_scales      (K, 3)   per-axis log-scale (anisotropic)
#      quaternions     (K, 4)   orientation
#      log_intensities (K,)     signal amplitude — NO opacity
#
#  Training loop:
#      One epoch = forward + backward over ALL M projection views,
#      then one optimizer.step().
#      This directly implements:
#          L = Σ_{k=1}^{M} ‖ I_k − MIP_splat_k ‖²
# ===================================================================
class MIPSplattingTrainer:
    """
    Train intensity-only 3D Gaussians against multi-view MIP ground truth.

    Replaces opacity with intensity; replaces alpha-compositing with MIP.
    """

    def __init__(
        self,
        means:           torch.Tensor,
        log_scales:      torch.Tensor,
        quaternions:     torch.Tensor,
        log_intensities: torch.Tensor,
        lr:              float                  = 1e-3,
        lambda_ssim:     float                  = 0.2,
        lambda_scale:    float                  = 0.001,
        scale_min:       float                  = 0.005,
        beta_mip:        float                  = 10.0,
        aspect_scales:   Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        means           : (K, 3)
        log_scales      : (K, 3) or (K,)/(K,1) — auto-upgraded to (K,3)
        quaternions     : (K, 4)
        log_intensities : (K,)  replaces log_amplitudes; no opacity
        lr              : Adam learning rate
        lambda_ssim     : SSIM loss weight
        lambda_scale    : scale regularisation weight
        scale_min       : penalise scales below this value
        beta_mip        : soft-MIP sharpness (higher → harder MIP)
        aspect_scales   : (3,) aspect correction from compute_aspect_scales()
        """
        self.device      = means.device
        self.lambda_ssim  = lambda_ssim
        self.lambda_scale = lambda_scale
        self.scale_min    = scale_min
        self.beta_mip     = beta_mip
        self.aspect_scales = aspect_scales

        # Auto-upgrade isotropic → anisotropic log_scales
        if log_scales.ndim == 1 or (log_scales.ndim == 2 and log_scales.shape[1] == 1):
            ls = log_scales.reshape(-1, 1).expand(-1, 3).clone().contiguous()
            print(f"  [MIPSplattingTrainer] log_scales upgraded "
                  f"{log_scales.shape} → {ls.shape}  (anisotropic)")
        else:
            ls = log_scales.clone()

        self.means           = nn.Parameter(means.clone())
        self.log_scales      = nn.Parameter(ls)                        # (K, 3)
        self.quaternions     = nn.Parameter(quaternions.clone())
        self.log_intensities = nn.Parameter(log_intensities.clone())   # (K,)

        self.optimizer = torch.optim.Adam([
            {'params': [self.means],           'lr': lr},
            {'params': [self.log_scales],      'lr': lr * 0.5},
            {'params': [self.quaternions],     'lr': lr * 0.3},
            {'params': [self.log_intensities], 'lr': lr},
        ])

        # Pruning
        self.prune_intens_thresh = 0.01
        self.prune_min_gaussians = 2000

    # ------------------------------------------------------------------
    #  Parameter construction
    # ------------------------------------------------------------------
    def _build_gaussians(self) -> GaussianParameters:
        """
        Construct GaussianParameters from learnable parameters.

        Σ_k = R_k · diag(s_k²) · R_kᵀ   (anisotropic, per-axis scales)
        intensity_k = sigmoid(log_intensities_k) ∈ (0, 1)
        """
        K = self.means.shape[0]

        # Per-axis scales (K, 3)
        scales = torch.exp(self.log_scales).clamp(1e-5, 1e2)

        # Quaternion → rotation matrix (K, 3, 3)
        q = F.normalize(self.quaternions, p=2, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        Rm = torch.zeros(K, 3, 3, device=self.device, dtype=q.dtype)
        Rm[:, 0, 0] = 1 - 2*(y*y + z*z);  Rm[:, 0, 1] = 2*(x*y - w*z);  Rm[:, 0, 2] = 2*(x*z + w*y)
        Rm[:, 1, 0] = 2*(x*y + w*z);      Rm[:, 1, 1] = 1 - 2*(x*x+z*z);Rm[:, 1, 2] = 2*(y*z - w*x)
        Rm[:, 2, 0] = 2*(x*z - w*y);      Rm[:, 2, 1] = 2*(y*z + w*x);  Rm[:, 2, 2] = 1 - 2*(x*x+y*y)

        # Anisotropic covariance Σ = R diag(s²) Rᵀ
        S2  = torch.diag_embed(scales ** 2)                     # (K, 3, 3)
        cov = Rm @ S2 @ Rm.transpose(-2, -1)                    # (K, 3, 3)

        # Intensity via sigmoid: (0, 1) — no opacity
        intensities = torch.sigmoid(self.log_intensities)        # (K,)

        return GaussianParameters(
            means       = self.means,
            covariances = cov,
            intensities = intensities,
        )

    def _build_gaussians_corrected(self) -> GaussianParameters:
        g = self._build_gaussians()
        if self.aspect_scales is not None:
            g = apply_aspect_correction(g, self.aspect_scales)
        return g

    # ------------------------------------------------------------------
    #  Single-projection forward + backward
    # ------------------------------------------------------------------
    def _forward_projection(
        self,
        camera:    Camera,
        gt_mip:    torch.Tensor,
        R:         torch.Tensor,
        T:         torch.Tensor,
        gaussians: GaussianParameters,         # ← passed in, built once per epoch
    ) -> dict:
        """
        Render one projection, compute loss, backpropagate.

        Loss_k = MSE(pred_k, gt_k) + λ_ssim·(1 − SSIM_k) + scale_reg

        NOTE: gaussians is passed in pre-built so the covariance computation
        (quaternion → R → Σ = R diag(s²) Rᵀ) runs once per epoch, not once
        per projection view.  optimizer.zero_grad() and optimizer.step() are
        managed externally by train_epoch().
        """
        pred_mip, n_vis = render_mip_projection(
            gaussians, camera, R, T, beta=self.beta_mip)    # (H, W)

        mse  = mse_loss(pred_mip, gt_mip)
        psnr = -10.0 * torch.log10(mse.clamp(min=1e-12))

        ssim_term = self.lambda_ssim * ssim_loss_fn(pred_mip, gt_mip)

        scales    = torch.exp(self.log_scales).clamp(1e-5, 1e2)
        scale_reg = self.lambda_scale * torch.clamp(
            self.scale_min - scales, min=0.0).mean()

        loss = mse + ssim_term + scale_reg
        loss.backward()

        # Free the rendered image immediately — no longer needed after backward
        del pred_mip

        return {
            'loss':      loss.item(),
            'mse':       mse.item(),
            'psnr':      psnr.item(),
            'ssim_term': ssim_term.item(),
            'scale_reg': scale_reg.item(),
            'n_visible': n_vis,
        }

    # ------------------------------------------------------------------
    #  Epoch: accumulate gradients over ALL M projections, then step
    # ------------------------------------------------------------------
    def train_epoch(
        self,
        camera:  Camera,
        dataset: List[dict],
    ) -> dict:
        """
        One epoch = forward+backward over ALL M projection views,
        then one optimizer.step().

        Gaussians are built ONCE per epoch (not per view) — the
        quaternion→R→Σ computation is shared across all M projections.

        Implements:  L = Σ_{k=1}^{M} ‖ I_k − MIP_splat_k ‖²
        """
        self.optimizer.zero_grad()

        # Build covariances once — O(K) matrix ops, not O(K×M)
        gaussians   = self._build_gaussians_corrected()

        all_metrics = []
        for view in dataset:
            m = self._forward_projection(
                camera,
                view['image'],
                view['R'],
                view['T'],
                gaussians,           # ← shared, not rebuilt
            )
            all_metrics.append(m)

        torch.nn.utils.clip_grad_norm_(
            [self.means, self.log_scales,
             self.quaternions, self.log_intensities],
            max_norm=1.0,
        )
        self.optimizer.step()

        with torch.no_grad():
            self.log_scales.data.clamp_(-9.0, 0.0)
            self.log_intensities.data.clamp_(-5.0, 5.0)
            self.means.data.clamp_(-1.0, 1.0)

        # Return fragmented GPU memory to allocator after full epoch
        if self.means.device.type == 'cuda':
            torch.cuda.empty_cache()

        return {k: float(np.mean([m[k] for m in all_metrics]))
                for k in all_metrics[0]}

    # ------------------------------------------------------------------
    #  Pruning
    # ------------------------------------------------------------------
    def prune_gaussians(self, epoch: int = 0) -> int:
        """Remove Gaussians with intensity below threshold."""
        with torch.no_grad():
            intens   = torch.sigmoid(self.log_intensities)
            keep     = intens > self.prune_intens_thresh
            n_before = keep.shape[0]
            n_keep   = int(keep.sum().item())

            if n_keep >= n_before or n_keep < self.prune_min_gaussians:
                return 0

            self.means           = nn.Parameter(self.means.data[keep].clone())
            self.log_scales      = nn.Parameter(self.log_scales.data[keep].clone())
            self.quaternions     = nn.Parameter(self.quaternions.data[keep].clone())
            self.log_intensities = nn.Parameter(self.log_intensities.data[keep].clone())

            lr = self.optimizer.param_groups[0]['lr']
            self.optimizer = torch.optim.Adam([
                {'params': [self.means],           'lr': lr},
                {'params': [self.log_scales],      'lr': lr * 0.5},
                {'params': [self.quaternions],     'lr': lr * 0.3},
                {'params': [self.log_intensities], 'lr': lr},
            ])
            n_pruned = n_before - n_keep
            print(f"  [Prune @ epoch {epoch}] {n_before} → {n_keep} "
                  f"(removed {n_pruned}, intensity < {self.prune_intens_thresh})")
            return n_pruned

    # ------------------------------------------------------------------
    #  Full training loop
    # ------------------------------------------------------------------
    def train(
        self,
        camera:      Camera,
        dataset:     List[dict],
        n_epochs:    int           = 200,
        log_every:   int           = 10,
        prune_every: int           = 25,
        save_path:   Optional[str] = None,
        save_every:  int           = 50,
    ) -> List[dict]:
        """
        Train for n_epochs. Each epoch covers all M projection views.

        Parameters
        ----------
        camera      : Camera
        dataset     : M projection dicts from generate_mip_dataset()
        n_epochs    : number of training epochs
        log_every   : print metrics every N epochs
        prune_every : prune dim Gaussians every N epochs (0 = disabled)
        save_path   : checkpoint template e.g. 'ckpt/ep{epoch}.pt'
        save_every  : save checkpoint every N epochs

        Returns
        -------
        history : list of per-epoch metric dicts
        """
        M       = len(dataset)
        history = []

        print(f"\nNeuroSGM — MIP Splatting Training")
        print(f"  Epochs       : {n_epochs}")
        print(f"  Projections  : M = {M}  (all iterated per epoch)")
        print(f"  Gaussians    : K = {self.means.shape[0]}")
        print(f"  log_scales   : {self.log_scales.shape}  (anisotropic)")
        print(f"  lambda_ssim  : {self.lambda_ssim}")
        print(f"  beta_mip     : {self.beta_mip}  (soft-MIP sharpness)")
        print("-" * 60)

        for epoch in range(1, n_epochs + 1):

            if prune_every > 0 and epoch % prune_every == 0:
                self.prune_gaussians(epoch)

            metrics = self.train_epoch(camera, dataset)
            history.append(metrics)

            if epoch % log_every == 0:
                print(f"  Epoch {epoch:>4d}/{n_epochs}  |  "
                      f"mse={metrics['mse']:.6f}  "
                      f"psnr={metrics['psnr']:.2f} dB  "
                      f"ssim_loss={metrics['ssim_term']:.4f}  "
                      f"K={self.means.shape[0]}")

            if save_path and epoch % save_every == 0:
                self._save_checkpoint(save_path.format(epoch=epoch), epoch)

        if save_path:
            self._save_checkpoint(save_path.format(epoch=n_epochs), n_epochs)

        return history

    def _save_checkpoint(self, path: str, epoch: int) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'means':           self.means.data.cpu(),
            'log_scales':      self.log_scales.data.cpu(),
            'quaternions':     self.quaternions.data.cpu(),
            'log_intensities': self.log_intensities.data.cpu(),
            'epoch':           epoch,
        }, path)
        print(f"  Checkpoint → {path}")


# ===================================================================
#  Analysis utilities
# ===================================================================
def save_training_analysis(
    history:            List[dict],
    validation_metrics: List[dict],
    out_dir:            str,
    validation_renders: Optional[List[dict]] = None,
) -> None:
    """Save CSV logs and PNG plots of training / validation results."""
    os.makedirs(out_dir, exist_ok=True)

    if history:
        keys = sorted({k for h in history for k in h.keys()})
        with open(os.path.join(out_dir, "training_history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch"] + keys)
            w.writeheader()
            for i, h in enumerate(history, 1):
                w.writerow({"epoch": i, **h})

    if validation_metrics:
        keys = sorted({k for v in validation_metrics for k in v.keys()})
        with open(os.path.join(out_dir, "validation_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(validation_metrics)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib unavailable ({exc})"); return

    if history:
        epochs = np.arange(1, len(history) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(epochs, [h['mse']        for h in history], color='tab:blue')
        axes[0].set_title('MSE Loss');  axes[0].set_xlabel('epoch')
        axes[1].plot(epochs, [h['psnr']       for h in history], color='tab:green')
        axes[1].set_title('PSNR (dB)'); axes[1].set_xlabel('epoch')
        ssim_curve = [h.get('ssim_term', 1.0 - h.get('ssim', 0.0)) for h in history]
        axes[2].plot(epochs, ssim_curve, color='tab:orange', label='ssim')
        axes[2].plot(epochs, [h['scale_reg']  for h in history], color='tab:red',    label='scale')
        axes[2].set_title('Regularisers'); axes[2].set_xlabel('epoch'); axes[2].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "training_metrics.png"), dpi=160)
        plt.close(fig)

    if validation_renders:
        n = min(6, len(validation_renders))
        fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.0))
        if n == 1:
            axes = np.array(axes).reshape(2, 1)
        for i in range(n):
            item = validation_renders[i]
            axes[0, i].imshow(item['gt'],   cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f"GT {item.get('label', i)}")
            axes[0, i].axis('off')
            axes[1, i].imshow(item['pred'], cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f"Pred {item.get('label', i)}")
            axes[1, i].axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "validation_renders.png"), dpi=180)
        plt.close(fig)


# ===================================================================
#  Main
# ===================================================================
if __name__ == "__main__":
    print("NeuroSGM — MIP-supervised Gaussian Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load volume
    vol_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "dataset",
        "10-2900-control-cell-05_cropped_corrected.tif"))
    print(f"\nLoading volume: {vol_path}")
    vol_np  = load_volume(vol_path)
    Z, Y, X = vol_np.shape
    vol_gpu = torch.from_numpy(vol_np).to(device)
    print(f"  Shape (Z,Y,X): ({Z},{Y},{X})")

    aspect_scales = compute_aspect_scales((Z, Y, X))
    print(f"  Aspect scales (x,y,z): {aspect_scales.tolist()}")

    # 2. Load Gaussian checkpoint
    ckpt_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "gmf_refined_best.pt"))
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    means           = ckpt["means"]
    log_scales      = ckpt["log_scales"]
    quaternions     = ckpt["quaternions"]
    # Accept old 'log_amplitudes' key or new 'log_intensities'
    log_intensities = ckpt.get("log_intensities", ckpt.get("log_amplitudes"))
    if "log_amplitudes" in ckpt:
        cuda_log_amplitudes = ckpt["log_amplitudes"]
    else:
        cuda_log_amplitudes = torch.log(torch.sigmoid(log_intensities).clamp(min=1e-6))
    print(f"  {means.shape[0]} Gaussians  |  log_scales: {log_scales.shape}")

    # 3. Camera
    H, W   = int(Y), int(X)
    camera = Camera.from_fov(fov_x_deg=50.0, width=W, height=H,
                             near=0.01, far=10.0)
    print(f"\nCamera: {W}×{H}  fx={camera.fx:.1f}")

    # 4. Generate GT MIP dataset  (M projection views)
    print("\nGenerating camera poses...")
    poses = generate_camera_poses(
        n_azimuth=20, n_elevation=10,
        elevation_range=(-60.0, 60.0),
        radius=3.5, include_axis_aligned=True,
    )
    print(f"  M = {len(poses)} projection views")

    print("\nRendering GT MIP dataset...")
    dataset = generate_mip_dataset(
        vol_gpu, camera, poses,
        n_ray_samples=200, near=0.5, far=6.0,
        aspect_scales=aspect_scales,
    )

    # 5. Train
    env_vps = os.getenv("HISNEGS_VIEWS_PER_STEP")
    try:
        views_per_step = int(env_vps) if env_vps else 16
    except ValueError:
        views_per_step = 16

    if CUDASplattingTrainer is not None:
        print("\nInitialising CUDASplattingTrainer...")
        trainer = CUDASplattingTrainer(
            means=means,
            log_scales=log_scales,
            quaternions=quaternions,
            log_amplitudes=cuda_log_amplitudes,
            aspect_scales=aspect_scales,
            lr=1e-3,
            pixels_per_step=8192,
            sampling_mode="tile",
            max_visible_gaussians=4096,
        )

        save_tmpl = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "checkpoints",
            "mip_ckpt", "splat_step{step}.pt"))

        history = trainer.train(
            camera=camera,
            dataset=dataset,
            n_steps=10000,
            log_every=100,
            views_per_step=views_per_step,
            save_path=save_tmpl,
            save_every=2000,
        )
    else:
        print("\nInitialising MIPSplattingTrainer... (CUDA trainer unavailable)")
        trainer = MIPSplattingTrainer(
            means           = means,
            log_scales      = log_scales,
            quaternions     = quaternions,
            log_intensities = log_intensities,
            lr              = 1e-3,
            lambda_ssim     = 0.2,
            lambda_scale    = 0.001,
            scale_min       = 0.005,
            beta_mip        = 10.0,
            aspect_scales   = aspect_scales,
        )

        save_tmpl = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "checkpoints",
            "mip_ckpt", "splat_ep{epoch}.pt"))

        history = trainer.train(
            camera      = camera,
            dataset     = dataset,
            n_epochs    = 200,
            log_every   = 10,
            prune_every = 25,
            save_path   = save_tmpl,
            save_every  = 50,
        )

    # 6. Validation
    print("\nValidation renders...")
    lpips_model = None
    try:
        import lpips as _lpips
        lpips_model = _lpips.LPIPS(net='alex').to(device).eval()
    except Exception as exc:
        print(f"  LPIPS unavailable: {exc}")

    val_metrics, val_renders = [], []
    gaussians = trainer._build_gaussians_corrected()
    if not hasattr(gaussians, "intensities") and hasattr(gaussians, "weights"):
        gaussians = GaussianParameters(
            means=gaussians.means,
            covariances=gaussians.covariances,
            intensities=gaussians.weights,
        )
    beta_mip = getattr(trainer, "beta_mip", 10.0)

    for vi in [0, len(dataset)//4, len(dataset)//2, len(dataset)-1]:
        view     = dataset[vi]
        pred_mip, n_vis = render_mip_projection(
            gaussians, camera, view['R'], view['T'],
            beta=beta_mip,
        )
        gt      = view['image']
        p_score = psnr_metric(pred_mip, gt)
        s_score = float(1.0 - ssim_loss_fn(pred_mip, gt).item())
        lp      = lpips_metric(pred_mip, gt, lpips_model)

        val_metrics.append({
            'view_idx':  vi,
            'elevation': view['elevation'],
            'azimuth':   view['azimuth'],
            'psnr':      p_score,
            'ssim':      s_score,
            'lpips':     lp,
            'n_visible': n_vis,
        })
        val_renders.append({
            'view_idx': vi,
            'label':    f"el={view['elevation']:.0f}°",
            'gt':       gt.detach().float().clamp(0, 1).cpu().numpy(),
            'pred':     pred_mip.detach().float().clamp(0, 1).cpu().numpy(),
        })
        print(f"  View {vi:>3d}  PSNR={p_score:.2f} dB  "
              f"SSIM={s_score:.4f}  LPIPS={lp:.4f}  visible={n_vis}")

    figure_dir = os.path.join(os.path.dirname(__file__), "figure")
    save_training_analysis(history, val_metrics, figure_dir,
                           validation_renders=val_renders)
    print(f"\nSaved analysis → {figure_dir}")
    print("\n✓ NeuroSGM MIP training complete!")