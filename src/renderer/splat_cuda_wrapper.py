"""
CUDA-accelerated Gaussian Splatting for Training
=================================================

Fuses the N×K inner splatting loop in CUDA while using PyTorch autograd
for the projection chain (quaternion→rotation→covariance→camera→2D).

Architecture:
  PyTorch autograd:  means, log_scales, quaternions, log_amplitudes
                     → covariances, opacities
                     → camera transform → 2D means, 2D covariance
                     → invert 2×2 → cov_inv (K,3)
  CUDA kernel:       cov_inv, means_2d, opacities, colors, pixels
                     → rendered (N,)   [forward]
                     → grad_means_2d, grad_cov_inv, grad_opacities, grad_colors [backward]
  PyTorch autograd:  chain rule back through projection to learnable params

This avoids materialising the (N, K) Gaussian evaluation tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple, Optional, List
import numpy as np
from types import SimpleNamespace

try:
    import splat_cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("WARNING: splat_cuda not found. Build with: python setup_splat_cuda.py build_ext --inplace")


class SplatAlphaFunction(Function):
    """
    Custom autograd function wrapping the CUDA splatting kernel.

    Forward: (means_2d, cov_inv, opacities, colors, pixels) → rendered
    Backward: grad_rendered → (grad_means_2d, grad_cov_inv, grad_opacities, grad_colors, None)
    """

    @staticmethod
    def forward(ctx, means_2d, cov_inv, opacities, colors, pixels):
        """
        Parameters
        ----------
        means_2d : (K, 2) float32 — sorted by depth
        cov_inv : (K, 3) float32 — [inv_a, inv_b, inv_d] sorted
        opacities : (K,) float32 — sorted
        colors : (K,) float32 — sorted (grayscale)
        pixels : (N, 2) float32

        Returns
        -------
        rendered : (N,) float32
        """
        rendered, T_out = splat_cuda.forward(
            means_2d, cov_inv, opacities, colors, pixels
        )
        ctx.save_for_backward(means_2d, cov_inv, opacities, colors, pixels)
        return rendered

    @staticmethod
    def backward(ctx, grad_rendered):
        means_2d, cov_inv, opacities, colors, pixels = ctx.saved_tensors

        grad_means_2d, grad_cov_inv, grad_opacities, grad_colors = splat_cuda.backward(
            grad_rendered.contiguous(),
            means_2d, cov_inv, opacities, colors, pixels
        )

        return grad_means_2d, grad_cov_inv, grad_opacities, grad_colors, None


def cuda_splat_alpha(means_2d, cov_inv, opacities, colors, pixels):
    """Differentiable alpha-compositing splatting via CUDA."""
    return SplatAlphaFunction.apply(means_2d, cov_inv, opacities, colors, pixels)


# ============================================================================
# Full differentiable pipeline: learnable params → rendered pixels
# ============================================================================

def build_covariances(quaternions, log_scales):
    """
    Build 3D covariance matrices from quaternions and log-scales.

    Parameters
    ----------
    quaternions : (K, 4)
    log_scales : (K, 3)

    Returns
    -------
    covariances : (K, 3, 3)
    """
    K = quaternions.shape[0]

    scales = torch.exp(log_scales).clamp(1e-5, 1e2)
    q = F.normalize(quaternions, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(K, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    S2 = torch.diag_embed(scales ** 2)
    covariances = R @ S2 @ R.transpose(-2, -1)

    return covariances


def transform_to_camera(means, covariances, R_cam, T_cam):
    """Transform Gaussians from world to camera frame."""
    means_cam = (R_cam @ means.unsqueeze(-1)).squeeze(-1) + T_cam.unsqueeze(0)
    cov_cam = R_cam.unsqueeze(0) @ covariances @ R_cam.T.unsqueeze(0)
    return means_cam, cov_cam


def project_to_2d(means_cam, cov_cam, fx, fy, cx, cy):
    """
    Project 3D Gaussians to 2D.

    Returns means_2d (K,2), cov_2d (K,2,2), depths (K,).
    """
    x = means_cam[:, 0]
    y = means_cam[:, 1]
    z = means_cam[:, 2]
    z_safe = z.clamp(min=1e-6)

    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy
    means_2d = torch.stack([u, v], dim=-1)

    # Jacobian
    z_sq = z_safe * z_safe
    K_g = means_cam.shape[0]
    J = torch.zeros(K_g, 2, 3, device=means_cam.device, dtype=means_cam.dtype)
    J[:, 0, 0] = fx / z_safe
    J[:, 0, 2] = -fx * x / z_sq
    J[:, 1, 1] = fy / z_safe
    J[:, 1, 2] = -fy * y / z_sq

    cov_2d = J @ cov_cam @ J.transpose(-2, -1)
    eps = 1e-4
    cov_2d = cov_2d + eps * torch.eye(2, device=cov_2d.device).unsqueeze(0)

    return means_2d, cov_2d, z


def invert_2x2(cov_2d):
    """
    Invert 2×2 symmetric matrices.

    Parameters
    ----------
    cov_2d : (K, 2, 2)

    Returns
    -------
    cov_inv_packed : (K, 3) — [inv_a, inv_b, inv_d]
    """
    a = cov_2d[:, 0, 0]
    b = cov_2d[:, 0, 1]
    d = cov_2d[:, 1, 1]

    det = a * d - b * b
    det_safe = det.clamp(min=1e-12)
    inv_det = 1.0 / det_safe

    inv_a = d * inv_det
    inv_b = -b * inv_det
    inv_d = a * inv_det

    return torch.stack([inv_a, inv_b, inv_d], dim=-1)


def cull_gaussians(means_cam, means_2d, cov_2d, near, far, width, height, radius_mult=3.0):
    """Visibility culling: depth + frustum."""
    z = means_cam[:, 2]
    depth_ok = (z > near) & (z < far)

    a = cov_2d[:, 0, 0]
    b = cov_2d[:, 0, 1]
    d = cov_2d[:, 1, 1]
    tr = a + d
    det = a * d - b * b
    disc = (tr * tr - 4.0 * det).clamp(min=0.0)
    lambda_max = 0.5 * (tr + torch.sqrt(disc))
    radius = radius_mult * torch.sqrt(lambda_max.clamp(min=1e-8))

    u = means_2d[:, 0]
    v = means_2d[:, 1]
    in_frustum = (
        (u + radius > 0) & (u - radius < width) &
        (v + radius > 0) & (v - radius < height)
    )

    return depth_ok & in_frustum


def apply_aspect_correction(means, covariances, aspect_scales):
    """Scale means and covariances by aspect ratios."""
    s = aspect_scales.to(means.device)
    means_c = means * s.unsqueeze(0)
    S = torch.diag(s)
    cov_c = S.unsqueeze(0) @ covariances @ S.unsqueeze(0)
    return means_c, cov_c


class CUDASplattingRenderer:
    """
    CUDA-accelerated Gaussian splatting renderer for training.

    Uses PyTorch autograd for projection chain + CUDA kernel for splatting.
    """

    def __init__(self, near=0.01, far=10.0, radius_mult=3.0, max_visible_gaussians=0, pixel_window_pad=8.0):
        self.near = near
        self.far = far
        self.radius_mult = radius_mult
        self.max_visible_gaussians = int(max_visible_gaussians)
        self.pixel_window_pad = float(pixel_window_pad)
        assert HAS_CUDA, "splat_cuda extension required. Build with setup_splat_cuda.py"

    def render_at_pixels(
        self,
        means, covariances, opacities, colors,
        R_cam, T_cam,
        fx, fy, cx, cy, width, height,
        pixels,
    ):
        """
        Render at specific pixel locations using CUDA splatting.

        Parameters
        ----------
        means : (K, 3) - world space (aspect-corrected)
        covariances : (K, 3, 3)
        opacities : (K,) - differentiable
        colors : (K,) - grayscale color (differentiable)
        R_cam, T_cam : camera extrinsics
        fx, fy, cx, cy : camera intrinsics
        width, height : image dimensions
        pixels : (N, 2) pixel coordinates

        Returns
        -------
        rendered : (N,) - differentiable w.r.t. all inputs
        """
        # Step 1: World → Camera (differentiable via autograd)
        means_cam, cov_cam = transform_to_camera(means, covariances, R_cam, T_cam)

        # Step 2: 3D → 2D projection (differentiable via autograd)
        means_2d, cov_2d, depths = project_to_2d(means_cam, cov_cam, fx, fy, cx, cy)

        # Visibility culling (no grad)
        with torch.no_grad():
            visible = cull_gaussians(
                means_cam, means_2d, cov_2d,
                self.near, self.far, width, height, self.radius_mult
            )

            px_min = pixels[:, 0].min() - self.pixel_window_pad
            px_max = pixels[:, 0].max() + self.pixel_window_pad
            py_min = pixels[:, 1].min() - self.pixel_window_pad
            py_max = pixels[:, 1].max() + self.pixel_window_pad

            a = cov_2d[:, 0, 0]
            b = cov_2d[:, 0, 1]
            d = cov_2d[:, 1, 1]
            tr = a + d
            det = a * d - b * b
            disc = (tr * tr - 4.0 * det).clamp(min=0.0)
            lambda_max = 0.5 * (tr + torch.sqrt(disc))
            radius = self.radius_mult * torch.sqrt(lambda_max.clamp(min=1e-8))

            u = means_2d[:, 0]
            v = means_2d[:, 1]
            overlap_window = (
                (u + radius >= px_min) & (u - radius <= px_max) &
                (v + radius >= py_min) & (v - radius <= py_max)
            )
            visible = visible & overlap_window

        n_vis = visible.sum().item()
        if n_vis == 0:
            return opacities.sum() * 0.0 + torch.zeros(
                pixels.shape[0], device=pixels.device)

        # Filter to visible
        means_2d_vis = means_2d[visible]
        cov_2d_vis = cov_2d[visible]
        opacities_vis = opacities[visible]
        colors_vis = colors[visible]
        depths_vis = depths[visible]

        if self.max_visible_gaussians > 0 and depths_vis.shape[0] > self.max_visible_gaussians:
            keep_idx = torch.topk(depths_vis, k=self.max_visible_gaussians, largest=False).indices
            means_2d_vis = means_2d_vis[keep_idx]
            cov_2d_vis = cov_2d_vis[keep_idx]
            opacities_vis = opacities_vis[keep_idx]
            colors_vis = colors_vis[keep_idx]
            depths_vis = depths_vis[keep_idx]

        # Sort by depth (differentiable: just reordering)
        order = torch.argsort(depths_vis)
        means_2d_sorted = means_2d_vis[order]
        cov_2d_sorted = cov_2d_vis[order]
        opacities_sorted = opacities_vis[order]
        colors_sorted = colors_vis[order]

        # Invert 2×2 covariance (differentiable via autograd)
        cov_inv = invert_2x2(cov_2d_sorted)  # (K_vis, 3)

        # Step 3+4: CUDA splatting (custom autograd backward)
        rendered = cuda_splat_alpha(
            means_2d_sorted, cov_inv, opacities_sorted, colors_sorted, pixels
        )

        return rendered


class CUDASplattingTrainer:
    """
    Training loop with CUDA-accelerated splatting and scale regularization.
    """

    def __init__(
        self,
        means, log_scales, quaternions, log_amplitudes,
        aspect_scales,
        lr=5e-4,
        pixels_per_step=16384,
        sampling_mode="tile",
        max_visible_gaussians=4096,
    ):
        self.device = means.device
        self.means = nn.Parameter(means.clone())
        self.log_scales = nn.Parameter(log_scales.clone())
        self.quaternions = nn.Parameter(quaternions.clone())
        self.log_amplitudes = nn.Parameter(log_amplitudes.clone())
        self.aspect_scales = aspect_scales.to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': [self.means], 'lr': lr},
            {'params': [self.log_scales], 'lr': lr * 0.5},
            {'params': [self.quaternions], 'lr': lr * 0.3},
            {'params': [self.log_amplitudes], 'lr': lr},
        ])

        self.pixels_per_step = pixels_per_step
        self.sampling_mode = sampling_mode
        self.renderer = CUDASplattingRenderer(max_visible_gaussians=max_visible_gaussians)

        # Regularization
        self.lambda_opacity = 0.0
        self.lambda_scale = 0.001
        self.scale_min_target = 0.005

        # Intensity-based pruning
        self.prune_every = 2000
        self.prune_intensity_thresh = 0.01
        self.prune_min_gaussians = 2000

    def _build_params(self):
        """Build covariances and intensity from learnable parameters."""
        covariances = build_covariances(self.quaternions, self.log_scales)
        intensity = torch.exp(self.log_amplitudes.clamp(-10.0, 6.0)).clamp(0.0, 1.0)
        return covariances, intensity

    def _build_gaussians_corrected(self):
        """Compatibility helper for rendering.py final validation path."""
        covariances, intensity = self._build_params()
        means = self.means

        if self.aspect_scales is not None:
            s = self.aspect_scales.to(means.device)
            means = means * s.unsqueeze(0)
            S = torch.diag(s)
            covariances = S.unsqueeze(0) @ covariances @ S.unsqueeze(0)

        return SimpleNamespace(
            means=means,
            covariances=covariances,
            weights=intensity,
            colors=intensity.unsqueeze(-1),
        )

    def train_step(
        self,
        camera,
        gt_image,
        R_cam,
        T_cam,
        precomputed: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        backward_scale: Optional[float] = None,
        retain_graph: bool = False,
    ):
        """
        One training step.

        Parameters
        ----------
        camera : Camera-like object with fx, fy, cx, cy, width, height
        gt_image : (H, W) ground truth
        R_cam, T_cam : camera extrinsics
        """
        H = int(camera.height)
        W = int(camera.width)

        # Build/reuse differentiable params
        if precomputed is None:
            covariances, intensity = self._build_params()
            means_c, cov_c = apply_aspect_correction(
                self.means, covariances, self.aspect_scales
            )
        else:
            means_c, cov_c, intensity = precomputed

        # Full projection pixels (no random pixel sampling)
        ys = torch.arange(H, device=self.device, dtype=torch.float32) + 0.5
        xs = torch.arange(W, device=self.device, dtype=torch.float32) + 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        pixels = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        gt_vals = gt_image.reshape(-1)

        # Opacity-free mode: use constant opacity for visibility, intensity in color
        opacities = torch.ones_like(intensity)
        colors = intensity

        # CUDA splatting render
        rendered = self.renderer.render_at_pixels(
            means_c, cov_c, opacities, colors,
            R_cam, T_cam,
            float(camera.fx), float(camera.fy),
            float(camera.cx), float(camera.cy),
            W, H, pixels,
        )

        # Projection reconstruction loss
        l1 = F.l1_loss(rendered, gt_vals)
        mse = F.mse_loss(rendered, gt_vals)

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_x = rendered.mean()
        mu_y = gt_vals.mean()
        sigma_x = ((rendered - mu_x) ** 2).mean()
        sigma_y = ((gt_vals - mu_y) ** 2).mean()
        sigma_xy = ((rendered - mu_x) * (gt_vals - mu_y)).mean()
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        )
        ssim = ssim.clamp(0.0, 1.0)

        psnr = -10.0 * torch.log10(mse.clamp(min=1e-12))

        opacity_reg = rendered.new_tensor(0.0)

        # Scale regularization
        scales_cur = torch.exp(self.log_scales).clamp(1e-5, 1e2)
        scale_penalty = torch.clamp(self.scale_min_target - scales_cur, min=0.0).mean()
        scale_reg = self.lambda_scale * scale_penalty

        loss = l1 + opacity_reg + scale_reg
        if backward_scale is not None:
            (loss * float(backward_scale)).backward(retain_graph=retain_graph)

        return {
            'loss': loss.item(),
            'l1': l1.item(),
            'mse': mse.item(),
            'psnr': psnr.item(),
            'ssim': ssim.item(),
            'opacity_reg': opacity_reg.item(),
            'scale_reg': scale_reg.item(),
        }

    def train(
        self,
        camera,
        dataset: List[dict],
        n_steps: int = 10000,
        log_every: int = 50,
        views_per_step: Optional[int] = 16,
        save_path: Optional[str] = None,
        save_every: int = 2000,
    ) -> List[dict]:
        n_views = len(dataset)
        history = []

        if views_per_step is None or views_per_step <= 0:
            n_views_step = n_views
        else:
            n_views_step = min(int(views_per_step), n_views)

        print(f"\nCUDA splatting training: {n_steps} steps, {n_views} views, "
              f"{n_views_step} views/step (full projection per view)")
        print(f"  Gaussians: {self.means.shape[0]}")
        print("-" * 60)

        for step in range(1, n_steps + 1):
            if self.prune_every > 0 and step % self.prune_every == 0 and step > 0:
                self.prune_gaussians(step)

            self.optimizer.zero_grad()
            per_view_loss = []
            per_view_mse = []
            per_view_l1 = []
            per_view_psnr = []
            per_view_ssim = []

            covariances, intensity = self._build_params()
            means_c, cov_c = apply_aspect_correction(
                self.means, covariances, self.aspect_scales
            )
            precomputed = (means_c, cov_c, intensity)

            view_order = torch.randperm(n_views, device=self.device)[:n_views_step].tolist()
            for view_idx in view_order:
                view = dataset[view_idx]
                is_last_view = (view_idx == view_order[-1])
                metrics_v = self.train_step(
                    camera,
                    view['image'],
                    view['R'],
                    view['T'],
                    precomputed=precomputed,
                    backward_scale=(1.0 / float(n_views_step)),
                    retain_graph=not is_last_view,
                )
                per_view_loss.append(float(metrics_v['loss']))

                per_view_mse.append(float(metrics_v['mse']))
                per_view_l1.append(float(metrics_v['l1']))
                per_view_psnr.append(float(metrics_v['psnr']))
                per_view_ssim.append(float(metrics_v['ssim']))

            torch.nn.utils.clip_grad_norm_(
                [self.means, self.log_scales, self.quaternions, self.log_amplitudes],
                max_norm=1.0,
            )
            self.optimizer.step()

            with torch.no_grad():
                self.log_scales.data.clamp_(-7.6, -1.2)
                self.log_amplitudes.data.clamp_(-9.2, 0.0)
                self.means.data.clamp_(-1.0, 1.0)

            metrics = {
                'loss': float(np.mean(per_view_loss)),
                'l1': float(np.mean(per_view_l1)),
                'mse': float(np.mean(per_view_mse)),
                'psnr': float(np.mean(per_view_psnr)),
                'ssim': float(np.mean(per_view_ssim)),
                'opacity_reg': 0.0,
                'scale_reg': float(self.lambda_scale * torch.clamp(
                    self.scale_min_target - torch.exp(self.log_scales).clamp(1e-5, 1e2),
                    min=0.0,
                ).mean().detach().item()),
            }
            history.append(metrics)

            if step % log_every == 0:
                avg_loss = float(np.mean([h['loss'] for h in history[-log_every:]]))
                avg_l1 = float(np.mean([h['l1'] for h in history[-log_every:]]))
                avg_psnr = float(np.mean([h['psnr'] for h in history[-log_every:]]))
                avg_ssim = float(np.mean([h['ssim'] for h in history[-log_every:]]))
                n_gauss = self.means.shape[0]
                print(f"  Step {step:>6d}/{n_steps}  |  loss={avg_loss:.6f}  "
                    f"l1={avg_l1:.6f}  psnr={avg_psnr:.2f}  ssim={avg_ssim:.4f}  K={n_gauss}")

            if save_path and step % save_every == 0:
                path = save_path.format(step=step)
                self.save_checkpoint(path, step)
                print(f"  Checkpoint saved → {path}")

        if save_path:
            path = save_path.format(step=n_steps)
            self.save_checkpoint(path, n_steps)
            print(f"  Final checkpoint → {path}")

        return history

    def prune_gaussians(self, step=0):
        """Remove low-intensity Gaussians."""
        with torch.no_grad():
            intensity = torch.exp(self.log_amplitudes.clamp(-10.0, 6.0)).clamp(0.0, 1.0)
            keep = intensity > self.prune_intensity_thresh
            n_before = keep.shape[0]
            n_keep = keep.sum().item()

            if n_keep >= n_before or n_keep < self.prune_min_gaussians:
                return 0

            self.means = nn.Parameter(self.means.data[keep].clone())
            self.log_scales = nn.Parameter(self.log_scales.data[keep].clone())
            self.quaternions = nn.Parameter(self.quaternions.data[keep].clone())
            self.log_amplitudes = nn.Parameter(self.log_amplitudes.data[keep].clone())

            lr = self.optimizer.param_groups[0]['lr']
            self.optimizer = torch.optim.Adam([
                {'params': [self.means], 'lr': lr},
                {'params': [self.log_scales], 'lr': lr * 0.5},
                {'params': [self.quaternions], 'lr': lr * 0.3},
                {'params': [self.log_amplitudes], 'lr': lr},
            ])

            n_pruned = n_before - n_keep
            print(f"  [Prune @ step {step}] {n_before} → {n_keep} "
                  f"(removed {n_pruned}, intensity < {self.prune_intensity_thresh})")
            return n_pruned

    def save_checkpoint(self, path, step):
        """Save learnable parameters."""
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'means': self.means.data.cpu(),
            'log_scales': self.log_scales.data.cpu(),
            'quaternions': self.quaternions.data.cpu(),
            'log_amplitudes': self.log_amplitudes.data.cpu(),
            'step': step,
        }, path)
