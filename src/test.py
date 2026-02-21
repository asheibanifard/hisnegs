#!/usr/bin/env python3
"""
test.py — Load neurogs_ckpt_2 checkpoints, plot training progression,
           and visualise the best model's results.

Pure-PyTorch evaluation (no custom CUDA extension needed).
"""
import os
import re
import math

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
import tifffile as tiff
from typing import Optional

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "neurogs_ckpt_2")
DATASET_DIR = os.path.join(SCRIPT_DIR, "..", "dataset")
TIF_PATH = os.path.join(DATASET_DIR, "10-2900-control-cell-05_cropped_corrected.tif")
CONFIG_PATH = os.path.join(CKPT_DIR, "config.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_tif(path: str) -> np.ndarray:
    vol = tiff.imread(path).astype(np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)


def extract_step(filename: str):
    m = re.search(r"step(\d+)", filename)
    return int(m.group(1)) if m else None


def compute_psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-12))


def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute SSIM for 2D grayscale images in [0,1]."""
    try:
        from skimage.metrics import structural_similarity as skimage_ssim
        return float(skimage_ssim(gt, pred, data_range=1.0))
    except Exception:
        # Fallback: global SSIM approximation if skimage is unavailable.
        gt_f = gt.astype(np.float64)
        pred_f = pred.astype(np.float64)
        mu_x = gt_f.mean()
        mu_y = pred_f.mean()
        sigma_x = gt_f.var()
        sigma_y = pred_f.var()
        sigma_xy = ((gt_f - mu_x) * (pred_f - mu_y)).mean()
        c1 = (0.01 * 1.0) ** 2
        c2 = (0.03 * 1.0) ** 2
        num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        return float(num / (den + 1e-12))


def _to_lpips_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    """(H,W) in [0,1] -> (1,3,H,W) in [-1,1]."""
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = t.repeat(1, 3, 1, 1)
    t = t * 2.0 - 1.0
    return t.to(device)


def compute_lpips(
    gt: np.ndarray,
    pred: np.ndarray,
    device: str = "cpu",
) -> Optional[float]:
    """Compute LPIPS; returns None if dependency is unavailable."""
    try:
        import lpips
    except Exception:
        return None

    with torch.no_grad():
        lpips_dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        metric = lpips.LPIPS(net="alex").to(lpips_dev).eval()
        gt_t = _to_lpips_tensor(gt, lpips_dev)
        pred_t = _to_lpips_tensor(pred, lpips_dev)
        score = metric(gt_t, pred_t).item()
    return float(score)


def fmt_metric(x: Optional[float], ndigits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "N/A"
    return f"{x:.{ndigits}f}"


def build_latex_table(rows: list[dict]) -> str:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Best model reconstruction metrics.}",
        "  \\label{tab:best_model_metrics}",
        "  \\begin{tabular}{lccccc}",
        "    \\toprule",
        "    Target & PSNR $\\uparrow$ & SSIM $\\uparrow$ & MSE $\\downarrow$ & MAE $\\downarrow$ & LPIPS $\\downarrow$ \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + f"{row['target']} & {fmt_metric(row['psnr'], 3)} & {fmt_metric(row['ssim'], 4)}"
            + f" & {fmt_metric(row['mse'], 6)} & {fmt_metric(row['mae'], 6)}"
            + f" & {fmt_metric(row['lpips'], 4)} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


# ── CUDA kernel support ────────────────────────────────────────────────────
try:
    import gaussian_eval_cuda
    from cuda_kernels import _build_L_chol
    HAS_CUDA_EXT = True
    print("✓ Custom CUDA extension loaded — using fast kernel path")
except ImportError:
    HAS_CUDA_EXT = False
    print("✗ CUDA extension not found — using PyTorch fallback")


# ── GMF evaluator (CUDA kernel when available, PyTorch fallback) ──────────

class GMFEvaluator:
    """Evaluate a Gaussian Mixture Field from a state_dict.
    Uses the fused CUDA kernel when available for ~2-3× speedup."""

    def __init__(self, state_dict: dict, device: str = "cpu"):
        self.means = state_dict["means"].to(device)
        self.log_scales = state_dict["log_scales"].to(device)
        self.quaternions = state_dict["quaternions"].to(device)
        self.log_amplitudes = state_dict["log_amplitudes"].to(device)
        self.device = device
        self.K = self.means.shape[0]

        # Pre-compute Cholesky and amplitudes once (inference only)
        with torch.no_grad():
            self._amps = torch.exp(self.log_amplitudes.clamp(-10.0, 6.0))
            s = torch.exp(self.log_scales).clamp(1e-5, 1e2)
            q = F.normalize(self.quaternions, p=2, dim=-1)
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            R = torch.zeros(self.K, 3, 3, device=device, dtype=q.dtype)
            R[:, 0, 0] = 1 - 2 * (y * y + z * z)
            R[:, 0, 1] = 2 * (x * y - w * z)
            R[:, 0, 2] = 2 * (x * z + w * y)
            R[:, 1, 0] = 2 * (x * y + w * z)
            R[:, 1, 1] = 1 - 2 * (x * x + z * z)
            R[:, 1, 2] = 2 * (y * z - w * x)
            R[:, 2, 0] = 2 * (x * z - w * y)
            R[:, 2, 1] = 2 * (y * z + w * x)
            R[:, 2, 2] = 1 - 2 * (x * x + y * y)
            S2 = torch.diag_embed(s ** 2)
            Sigma = R @ S2 @ R.transpose(-2, -1)
            eps = 1e-5
            Sigma_reg = Sigma + eps * torch.eye(3, device=device).unsqueeze(0)
            self._L_chol = torch.linalg.cholesky(Sigma_reg.float())

        self._use_cuda = HAS_CUDA_EXT and device == "cuda"

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, k_chunk: int = 512) -> torch.Tensor:
        """Evaluate field at points x (N,3) -> (N,)."""
        N = x.shape[0]

        if self._use_cuda:
            # Fused CUDA kernel: forward returns (N, K), sum over K
            vals_nk = gaussian_eval_cuda.forward(
                x.contiguous().float(),
                self.means.contiguous().float(),
                self._L_chol.contiguous().float(),
                self._amps.contiguous().float(),
            )
            return vals_nk.sum(dim=1).to(x.dtype)

        # PyTorch fallback (chunked)
        K = self.K
        L = self._L_chol
        amps = self._amps
        out = torch.zeros(N, device=x.device, dtype=x.dtype)
        for ks in range(0, K, k_chunk):
            ke = min(ks + k_chunk, K)
            G = ke - ks
            mu = self.means[ks:ke]
            a = amps[ks:ke]
            Lc = L[ks:ke]

            diff = x[:, None, :] - mu[None, :, :]           # (N, G, 3)
            diff_flat = diff.reshape(N * G, 3, 1).float()
            L_exp = Lc.unsqueeze(0).expand(N, G, 3, 3).reshape(N * G, 3, 3)
            y = torch.linalg.solve_triangular(L_exp, diff_flat, upper=False)
            mahal = (y.squeeze(-1) ** 2).sum(-1).reshape(N, G)
            vals = a[None, :] * torch.exp(-0.5 * mahal)
            out = out + vals.sum(dim=1)
        return out


# ══════════════════════════════════════════════════════════════════════════
#  1. Discover & load checkpoints
# ══════════════════════════════════════════════════════════════════════════
print("Loading config ...")
cfg = load_config(CONFIG_PATH)
print("Loading volume ...")
vol = load_tif(TIF_PATH)
vol_gpu = torch.from_numpy(vol).float()
print(f"  Volume shape: {vol.shape}")

# Ground-truth MIP (z-axis maximum intensity projection)
gt_mip = vol.max(axis=0)  # (Y, X)

# Discover checkpoint files
ckpt_files = sorted(
    [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt") and "step" in f],
    key=lambda f: extract_step(f),
)
best_file = os.path.join(CKPT_DIR, "gmf_refined_best.pt")
final_file = os.path.join(CKPT_DIR, "gmf_refined.pt")

print(f"Found {len(ckpt_files)} step checkpoints + best + final")

# ══════════════════════════════════════════════════════════════════════════
#  2. Extract parameter statistics over training
# ══════════════════════════════════════════════════════════════════════════
print("\nExtracting training statistics ...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Using device: {device}")
Z, Y, X = vol.shape

# Prepare evaluation points (fixed random subset for consistency)
torch.manual_seed(42)
eval_n = min(100_000, vol_gpu.numel())
idx = torch.randint(0, vol_gpu.numel(), (eval_n,))
ez = idx // (Y * X)
ey = (idx % (Y * X)) // X
ex = idx % X
eval_pts = torch.stack([
    (ex.float() / max(X - 1, 1)) * 2 - 1,
    (ey.float() / max(Y - 1, 1)) * 2 - 1,
    (ez.float() / max(Z - 1, 1)) * 2 - 1,
], dim=1).to(device)
eval_gt = vol_gpu[ez, ey, ex].to(device)

# Containers for stats
steps_list = []
psnr_list = []
mse_list = []
mae_list = []
num_gaussians_list = []
mean_scale_list = []
std_scale_list = []
mean_amp_list = []
std_amp_list = []

for ci, fname in enumerate(ckpt_files):
    step = extract_step(fname)
    if step is None:
        continue
    path = os.path.join(CKPT_DIR, fname)
    state = torch.load(path, map_location="cpu", weights_only=True)
    K = state["means"].shape[0]

    # Parameter stats (on CPU)
    scales = torch.exp(state["log_scales"]).clamp(1e-5, 1e2)
    amps = torch.exp(state["log_amplitudes"].clamp(-10, 6))

    steps_list.append(step)
    num_gaussians_list.append(K)
    mean_scale_list.append(float(scales.mean()))
    std_scale_list.append(float(scales.std()))
    mean_amp_list.append(float(amps.mean()))
    std_amp_list.append(float(amps.std()))

    # Evaluate PSNR on this checkpoint
    evaluator = GMFEvaluator(state, device=device)
    pred_chunks = []
    for ci2 in range(0, eval_n, 8192):
        pred_chunks.append(evaluator(eval_pts[ci2:ci2 + 8192]))
    pred = torch.cat(pred_chunks)
    mse = F.mse_loss(pred, eval_gt).item()
    mae_val = F.l1_loss(pred, eval_gt).item()
    psnr = -10 * math.log10(max(mse, 1e-12))

    psnr_list.append(psnr)
    mse_list.append(mse)
    mae_list.append(mae_val)

    print(f"  Step {step:>6d}: K={K:>6d}  PSNR={psnr:.2f} dB  MSE={mse:.6f}  "
          f"scale={scales.mean():.4f}  amp={amps.mean():.4f}")

    del evaluator
    if device == "cuda":
        torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════════════════
#  3. Plot training curves
# ══════════════════════════════════════════════════════════════════════════
print("\nPlotting training curves ...")

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Training Progression -- neurogs_ckpt_2", fontsize=16, fontweight="bold")

steps_k = [s / 1000 for s in steps_list]

# (a) PSNR
ax = axes[0, 0]
ax.plot(steps_k, psnr_list, "b-o", markersize=3, linewidth=1.5)
best_idx = int(np.argmax(psnr_list))
ax.axvline(steps_k[best_idx], color="r", ls="--", alpha=0.6,
           label=f"Best: {psnr_list[best_idx]:.2f} dB @ step {steps_list[best_idx]}")
ax.set_xlabel("Step (x1000)")
ax.set_ylabel("PSNR (dB)")
ax.set_title("PSNR")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (b) MSE
ax = axes[0, 1]
ax.semilogy(steps_k, mse_list, "r-o", markersize=3, linewidth=1.5)
ax.set_xlabel("Step (x1000)")
ax.set_ylabel("MSE (log)")
ax.set_title("MSE")
ax.grid(True, alpha=0.3)

# (c) Number of Gaussians
ax = axes[1, 0]
ax.plot(steps_k, num_gaussians_list, "g-o", markersize=3, linewidth=1.5)
ax.set_xlabel("Step (x1000)")
ax.set_ylabel("# Gaussians")
ax.set_title("Number of Gaussians (densify/prune)")
ax.grid(True, alpha=0.3)

# (d) Mean scale
ax = axes[1, 1]
ax.plot(steps_k, mean_scale_list, "m-o", markersize=3, linewidth=1.5, label="mean")
ax.fill_between(
    steps_k,
    [m - s for m, s in zip(mean_scale_list, std_scale_list)],
    [m + s for m, s in zip(mean_scale_list, std_scale_list)],
    alpha=0.2, color="m",
)
ax.set_xlabel("Step (x1000)")
ax.set_ylabel("Scale")
ax.set_title("Mean Gaussian Scale (+/-1 std)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (e) Mean amplitude
ax = axes[2, 0]
ax.plot(steps_k, mean_amp_list, "c-o", markersize=3, linewidth=1.5, label="mean")
ax.fill_between(
    steps_k,
    [m - s for m, s in zip(mean_amp_list, std_amp_list)],
    [m + s for m, s in zip(mean_amp_list, std_amp_list)],
    alpha=0.2, color="c",
)
ax.set_xlabel("Step (x1000)")
ax.set_ylabel("Amplitude")
ax.set_title("Mean Gaussian Amplitude (+/-1 std)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (f) MAE
ax = axes[2, 1]
ax.plot(steps_k, mae_list, "k-o", markersize=3, linewidth=1.5)
ax.set_xlabel("Step (x1000)")
ax.set_ylabel("MAE")
ax.set_title("Mean Absolute Error")
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
train_fig_path = os.path.join(OUTPUT_DIR, "training_curves.pdf")
plt.savefig(train_fig_path, dpi=150, bbox_inches="tight")
print(f"  Saved -> {train_fig_path}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
#  4. Load best model and show its results
# ══════════════════════════════════════════════════════════════════════════
print("\nLoading best model ...")
best_state = torch.load(best_file, map_location="cpu", weights_only=True)
K_best = best_state["means"].shape[0]
field_best = GMFEvaluator(best_state, device=device)
print(f"  Best model: K={K_best}")

# -- 4a. Full-volume PSNR -------------------------------------------------
pred_chunks = []
for ci in range(0, eval_n, 8192):
    pred_chunks.append(field_best(eval_pts[ci:ci + 8192]))
pred_full = torch.cat(pred_chunks)
mse_best = F.mse_loss(pred_full, eval_gt).item()
psnr_best = -10 * math.log10(max(mse_best, 1e-12))
print(f"  Best PSNR: {psnr_best:.2f} dB  MSE: {mse_best:.6f}")

# -- 4b. Reconstructed MIP (z-axis) ---------------------------------------
print("  Computing reconstructed MIP ...")
n_z_mip = Z
z_vals = torch.linspace(-1, 1, n_z_mip)
recon_mip = torch.zeros(Y, X)

for yi in range(Y):
    yn = (yi / max(Y - 1, 1)) * 2 - 1
    xn = torch.linspace(-1, 1, X)
    max_row = torch.zeros(X)
    for zi in range(n_z_mip):
        zn = z_vals[zi].item()
        pts = torch.stack([
            xn,
            torch.full((X,), yn),
            torch.full((X,), zn),
        ], dim=1).to(device)
        vals = field_best(pts).cpu()
        max_row = torch.max(max_row, vals)
    recon_mip[yi] = max_row
    if (yi + 1) % 100 == 0:
        print(f"    MIP row {yi+1}/{Y}")

recon_mip_np = recon_mip.numpy()

# -- 4c. Reconstructed volume slices --------------------------------------
print("  Computing volume slices ...")
mid_z = Z // 2
zn_mid = (mid_z / max(Z - 1, 1)) * 2 - 1

ys = torch.linspace(-1, 1, Y)
xs = torch.linspace(-1, 1, X)
yy, xx = torch.meshgrid(ys, xs, indexing="ij")
pts_xy = torch.stack([
    xx.reshape(-1),
    yy.reshape(-1),
    torch.full((Y * X,), zn_mid),
], dim=1).to(device)

recon_slice = []
for ci in range(0, Y * X, 8192):
    recon_slice.append(field_best(pts_xy[ci:ci + 8192]).cpu())
recon_slice = torch.cat(recon_slice).reshape(Y, X).numpy()
gt_slice = vol[mid_z]

# -- 4e. Image metrics + LaTeX table --------------------------------------
print("  Computing image metrics (MIP/slice) ...")

def compute_image_metrics(gt_img: np.ndarray, pred_img: np.ndarray, device: str) -> dict:
    gt_img = np.clip(gt_img.astype(np.float32), 0.0, 1.0)
    pred_img = np.clip(pred_img.astype(np.float32), 0.0, 1.0)
    mse = float(np.mean((gt_img - pred_img) ** 2))
    mae = float(np.mean(np.abs(gt_img - pred_img)))
    psnr = compute_psnr_from_mse(mse)
    ssim = compute_ssim(gt_img, pred_img)
    lpips_val = compute_lpips(gt_img, pred_img, device=device)
    return {
        "psnr": psnr,
        "ssim": ssim,
        "mse": mse,
        "mae": mae,
        "lpips": lpips_val,
    }


mip_metrics = compute_image_metrics(gt_mip, recon_mip_np, device=device)
slice_metrics = compute_image_metrics(gt_slice, recon_slice, device=device)

latex_rows = [
    {"target": "MIP (z-axis)", **mip_metrics},
    {"target": f"Mid-Z slice ($z={mid_z}$)", **slice_metrics},
]
latex_table = build_latex_table(latex_rows)
latex_table_path = os.path.join(OUTPUT_DIR, "best_model_metrics_table.tex")
with open(latex_table_path, "w", encoding="utf-8") as f:
    f.write(latex_table + "\n")

print("  Saved LaTeX table ->", latex_table_path)
print("\nLaTeX table preview:\n")
print(latex_table)

# -- 4d. Parameter distributions of best model ----------------------------
scales_best = torch.exp(best_state["log_scales"]).clamp(1e-5, 1e2).numpy()
amps_best = torch.exp(best_state["log_amplitudes"].clamp(-10, 6)).numpy()
means_best = best_state["means"].numpy()

# Convert Gaussian means from normalised [-1,1] back to voxel coordinates
# so scatter plots preserve the original physical proportions.
# Normalised coord c -> voxel = (c + 1) / 2 * (dim - 1)
means_voxel = np.zeros_like(means_best)
means_voxel[:, 0] = (means_best[:, 0] + 1) / 2 * (X - 1)  # x -> voxel-X
means_voxel[:, 1] = (means_best[:, 1] + 1) / 2 * (Y - 1)  # y -> voxel-Y
means_voxel[:, 2] = (means_best[:, 2] + 1) / 2 * (Z - 1)  # z -> voxel-Z

# ══════════════════════════════════════════════════════════════════════════
#  5. Plot best model results
# ══════════════════════════════════════════════════════════════════════════
print("\nPlotting best model results ...")

fig = plt.figure(figsize=(18, 22))
gs_outer = gridspec.GridSpec(4, 1, figure=fig, hspace=0.35,
                             height_ratios=[1, 1, 0.6, 1])
fig.suptitle(f"Best Model Results -- K={K_best}  PSNR={psnr_best:.2f} dB",
             fontsize=16, fontweight="bold")

# --- Row 0 (4 cols): MIP comparison ---
gs_row0 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[0], wspace=0.3)

ax1 = fig.add_subplot(gs_row0[0])
ax1.imshow(gt_mip, cmap="gray", aspect="equal")
ax1.set_title("Ground Truth MIP (z-axis)")
ax1.axis("off")

ax2 = fig.add_subplot(gs_row0[1])
ax2.imshow(recon_mip_np, cmap="gray", aspect="equal", vmin=0, vmax=1)
ax2.set_title("Reconstructed MIP (z-axis)")
ax2.axis("off")

mip_diff = gt_mip - recon_mip_np
vabs = max(abs(mip_diff.min()), abs(mip_diff.max()))

ax_mip_diff = fig.add_subplot(gs_row0[2])
im_diff = ax_mip_diff.imshow(mip_diff, cmap="bwr", aspect="equal",
                              vmin=-vabs, vmax=vabs)
ax_mip_diff.set_title(f"MIP Error (GT-Recon)\nmean={mip_diff.mean():.4f}")
ax_mip_diff.axis("off")
plt.colorbar(im_diff, ax=ax_mip_diff, fraction=0.046)

ax_mip_abs = fig.add_subplot(gs_row0[3])
mip_error = np.abs(mip_diff)
im_abs = ax_mip_abs.imshow(mip_error, cmap="hot", aspect="equal")
ax_mip_abs.set_title(f"MIP |Error| (mean={mip_error.mean():.4f})")
ax_mip_abs.axis("off")
plt.colorbar(im_abs, ax=ax_mip_abs, fraction=0.046)

# --- Row 1 (4 cols): Slice comparison ---
gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[1], wspace=0.3)

ax3 = fig.add_subplot(gs_row1[0])
ax3.imshow(gt_slice, cmap="gray", aspect="equal")
ax3.set_title(f"GT Slice (z={mid_z})")
ax3.axis("off")

ax4 = fig.add_subplot(gs_row1[1])
ax4.imshow(recon_slice, cmap="gray", aspect="equal", vmin=0, vmax=1)
ax4.set_title(f"Recon Slice (z={mid_z})")
ax4.axis("off")

slice_diff = gt_slice - recon_slice
vabs_s = max(abs(slice_diff.min()), abs(slice_diff.max()))

ax5 = fig.add_subplot(gs_row1[2])
im5 = ax5.imshow(slice_diff, cmap="bwr", aspect="equal",
                  vmin=-vabs_s, vmax=vabs_s)
ax5.set_title(f"Slice Error (GT-Recon)\nmean={slice_diff.mean():.4f}")
ax5.axis("off")
plt.colorbar(im5, ax=ax5, fraction=0.046)

ax6 = fig.add_subplot(gs_row1[3])
error_map = np.abs(slice_diff)
im6 = ax6.imshow(error_map, cmap="hot", aspect="equal")
ax6.set_title(f"Slice |Error| (mean={error_map.mean():.4f})")
ax6.axis("off")
plt.colorbar(im6, ax=ax6, fraction=0.046)

# --- Row 2 (2 cols): Parameter distributions ---
gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[2], wspace=0.3)

ax7 = fig.add_subplot(gs_row2[0])
ax7.hist(scales_best.flatten(), bins=100, color="purple", alpha=0.7,
         edgecolor="black", linewidth=0.3)
ax7.set_xlabel("Scale")
ax7.set_ylabel("Count")
ax7.set_title(f"Scale Distribution\nmean={scales_best.mean():.4f}")
ax7.set_yscale("log")
ax7.grid(True, alpha=0.3)

ax8 = fig.add_subplot(gs_row2[1])
ax8.hist(amps_best, bins=100, color="teal", alpha=0.7,
         edgecolor="black", linewidth=0.3)
ax8.set_xlabel("Amplitude")
ax8.set_ylabel("Count")
ax8.set_title(f"Amplitude Distribution\nmean={amps_best.mean():.4f}")
ax8.set_yscale("log")
ax8.grid(True, alpha=0.3)

# --- Row 3 (3 cols): Gaussian means scatter plots ---
gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[3], wspace=0.3)

ax9 = fig.add_subplot(gs_row3[0])
sc = ax9.scatter(
    means_voxel[:, 0], means_voxel[:, 1],
    c=amps_best, cmap="viridis", s=0.5, alpha=0.5,
)
ax9.set_xlabel("X (voxel)")
ax9.set_ylabel("Y (voxel)")
ax9.set_title("Gaussian Means (XY, coloured by amp)")
ax9.set_aspect("equal")
ax9.set_box_aspect(1)
plt.colorbar(sc, ax=ax9, fraction=0.046, label="Amplitude")

ax10 = fig.add_subplot(gs_row3[1])
sc2 = ax10.scatter(
    means_voxel[:, 0], means_voxel[:, 2],
    c=amps_best, cmap="viridis", s=0.5, alpha=0.5,
)
ax10.set_xlabel("X (voxel)")
ax10.set_ylabel("Z (voxel)")
ax10.set_title("Gaussian Means (XZ, coloured by amp)")
ax10.set_aspect("equal")
ax10.set_box_aspect(1)
plt.colorbar(sc2, ax=ax10, fraction=0.046, label="Amplitude")

ax11 = fig.add_subplot(gs_row3[2])
sc3 = ax11.scatter(
    means_voxel[:, 2], means_voxel[:, 1],
    c=amps_best, cmap="viridis", s=0.5, alpha=0.5,
)
ax11.set_xlabel("Z (voxel)")
ax11.set_ylabel("Y (voxel)")
ax11.set_title("Gaussian Means (ZY, coloured by amp)")
ax11.set_aspect("equal")
ax11.set_box_aspect(1)
plt.colorbar(sc3, ax=ax11, fraction=0.046, label="Amplitude")

best_fig_path = os.path.join(OUTPUT_DIR, "best_model_results.pdf")
plt.savefig(best_fig_path, dpi=150, bbox_inches="tight")
print(f"  Saved -> {best_fig_path}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Checkpoint dir : {os.path.abspath(CKPT_DIR)}")
print(f"  # checkpoints  : {len(ckpt_files)} steps + best + final")
print(f"  Volume shape   : {vol.shape}")
print(f"  Best model K   : {K_best}")
print(f"  Best PSNR      : {psnr_best:.2f} dB")
print(f"  Best MSE       : {mse_best:.6f}")
print(f"  Peak PSNR step : {steps_list[best_idx]}")
print(f"  Outputs:")
print(f"    {os.path.abspath(train_fig_path)}")
print(f"    {os.path.abspath(best_fig_path)}")
print(f"    {os.path.abspath(latex_table_path)}")
print("=" * 60)
