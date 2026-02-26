#!/usr/bin/env python3
"""
viser_viewer.py — Interactive MIP-rendered 3D Gaussian Viewer
=============================================================

Loads trained 3D Gaussians and renders **true MIP projections**
(not alpha-blending) from the camera viewpoint, re-rendered
live on every camera movement in the viser browser viewer.

Mouse-drag the viewport to orbit the neuron.
The server intercepts every camera update, converts the pose to
(elevation, azimuth, radius), re-renders MIP server-side,
and streams the result back as a per-client background image.

A lightweight point-cloud overlay provides instant 3D reference
while the MIP is being re-rendered.

Usage:
    cd /workspace/hisnegs/src/renderer
    python viser_viewer.py
    python viser_viewer.py --ckpt ../checkpoints/mip_ckpt/e2e_ep400.pt
    python viser_viewer.py --port 8080 --res 512
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import viser

# ── Imports from rendering.py ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rendering import (
    Camera,
    GaussianParameters,
    compute_aspect_scales,
    apply_aspect_correction,
    render_mip_projection,
    load_config,
    _orbit_pose,
)


# =====================================================================
#  Load & build Gaussian parameters on GPU
# =====================================================================
def load_gaussians(ckpt_path: str, device: torch.device):
    """Load checkpoint → (means, covariances, intensities) tensors."""
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Loaded {ckpt_path}")
    print(f"  Epoch {ckpt['epoch']},  K = {ckpt['means'].shape[0]}")

    means           = ckpt["means"].float().to(device)
    log_scales      = ckpt["log_scales"].float().to(device)
    quaternions     = ckpt["quaternions"].float().to(device)
    log_intensities = ckpt["log_intensities"].float().to(device)

    K = means.shape[0]

    # Covariance: Σ = R diag(s²) Rᵀ
    scales = torch.exp(log_scales).clamp(1e-5, 1e2)
    q = F.normalize(quaternions, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(K, 3, 3, device=device)
    R[:, 0, 0] = 1-2*(y*y+z*z); R[:, 0, 1] = 2*(x*y-w*z); R[:, 0, 2] = 2*(x*z+w*y)
    R[:, 1, 0] = 2*(x*y+w*z);   R[:, 1, 1] = 1-2*(x*x+z*z); R[:, 1, 2] = 2*(y*z-w*x)
    R[:, 2, 0] = 2*(x*z-w*y);   R[:, 2, 1] = 2*(y*z+w*x);   R[:, 2, 2] = 1-2*(x*x+y*y)

    S2  = torch.diag_embed(scales ** 2)
    cov = R @ S2 @ R.transpose(-2, -1)

    intensities = torch.sigmoid(log_intensities)

    return means, cov, intensities, ckpt["epoch"]


# =====================================================================
#  Render a single MIP frame → (H, W, 3) uint8 numpy
# =====================================================================
@torch.no_grad()
def render_mip_frame(
    gaussians: GaussianParameters,
    camera:    Camera,
    el_deg:    float,
    az_deg:    float,
    radius:    float,
    beta:      float,
    device:    torch.device,
) -> np.ndarray:
    """Render one MIP view → (H, W, 3) uint8 (gray→RGB for viser)."""
    R_cam, T_cam = _orbit_pose(el_deg, az_deg, radius)
    R_cam = R_cam.to(device)
    T_cam = T_cam.to(device)

    img, _ = render_mip_projection(
        gaussians, camera, R_cam, T_cam,
        beta=beta, chunk_size=4096,
    )
    # Normalise to [0, 255]
    img_np = img.cpu().numpy()
    vmax = img_np.max() + 1e-8
    img_np = np.clip(img_np / vmax, 0.0, 1.0)
    gray = (img_np * 255).astype(np.uint8)
    # Convert to RGB (H, W, 3) — required by set_background_image
    rgb = np.stack([gray, gray, gray], axis=-1)
    return rgb


# =====================================================================
#  Extract (elevation, azimuth, radius) from viser camera state
# =====================================================================
def camera_to_orbit(cam_handle) -> tuple[float, float, float]:
    """Convert viser camera position/look_at → (el_deg, az_deg, radius).

    Uses same Y-up convention as _orbit_pose():
        cam_pos = (r·cos(el)·sin(az),  r·sin(el),  r·cos(el)·cos(az))

    Clamps elevation to ±89° to avoid pole singularities where the
    cross-product in _orbit_pose() degenerates (forward ≈ ±Y).
    """
    pos  = np.array(cam_handle.position,  dtype=np.float64)
    look = np.array(cam_handle.look_at,   dtype=np.float64)
    rel  = pos - look                       # relative to look_at
    radius = float(np.linalg.norm(rel))
    if radius < 1e-6:
        return 0.0, 0.0, 3.5  # fallback

    # Elevation from Y component
    sin_el = np.clip(rel[1] / radius, -1.0, 1.0)
    el_deg = math.degrees(math.asin(sin_el))
    el_deg = np.clip(el_deg, -89.0, 89.0)      # avoid pole singularity

    # Azimuth from XZ plane — guard against cos(el)≈0 at poles
    cos_el = math.cos(math.radians(el_deg))
    if cos_el < 1e-6:
        az_deg = 0.0                            # degenerate; pick arbitrary
    else:
        az_deg = math.degrees(math.atan2(rel[0], rel[2]))
    return el_deg, az_deg, radius


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Interactive MIP Gaussian Viewer (viser)")
    parser.add_argument("--ckpt", default="../checkpoints/mip_ckpt/e2e_ep800.pt")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--res", type=int, default=256, help="MIP render resolution")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.ckpt))

    # ── Load model ──
    means, cov, intensities, epoch = load_gaussians(ckpt_path, device)
    K = means.shape[0]

    # ── Aspect correction ──
    vol_shape = (100, 647, 813)
    aspect_scales = compute_aspect_scales(vol_shape).to(device)
    gaussians = GaussianParameters(means=means, covariances=cov, intensities=intensities)
    gaussians = apply_aspect_correction(gaussians, aspect_scales)

    # ── Camera matching training config ──
    cfg = load_config(os.path.join(os.path.dirname(__file__), "config_splat.yml"))
    res = args.res
    cam_render = Camera.from_config(cfg, width=res, height=res)
    default_radius = cfg["poses"]["radius"]
    default_beta   = cfg["training"]["beta_mip"]



    # ── GPU render lock (one MIP at a time to avoid CUDA contention) ──
    render_lock = threading.Lock()

    # ── Viser server ──
    server = viser.ViserServer(port=args.port)

    # Initial camera pose
    server.initial_camera.position  = (0.0, 0.0, float(default_radius))
    server.initial_camera.look_at   = (0.0, 0.0, 0.0)
    server.initial_camera.fov       = math.radians(cfg["camera"]["fov_x_deg"])
    server.initial_camera.up_direction = (0.0, 1.0, 0.0)

    # ── GUI controls ──
    with server.gui.add_folder("MIP Settings"):
        beta_slider = server.gui.add_slider(
            "Beta (MIP sharpness)", min=1.0, max=100.0, step=1.0,
            initial_value=default_beta,
        )
        render_btn = server.gui.add_button("Force Re-render")

    # ── Per-client status label (via server-level fallback) ──
    status = server.gui.add_markdown(
        f"**Epoch {epoch}** | K={K} | {res}x{res} | Rotate the 3D view to explore"
    )

    # ── Render initial MIP (applies to first client) ──
    init_frame = render_mip_frame(
        gaussians, cam_render, 15.0, 0.0, default_radius, default_beta, device
    )
    server.scene.set_background_image(init_frame, format="jpeg", jpeg_quality=90)

    # ── Per-client camera tracking ──
    # Each client gets a camera.on_update callback that re-renders
    # MIP from the new viewpoint and sends back via
    # client.scene.set_background_image (per-client).

    # ── Per-client debounce state ──
    # Tracks the last requested render params per client so we can
    # skip stale requests and only render the latest camera pose.
    _pending_render: dict[int, dict] = {}      # client_id → {el, az, radius, time}
    _last_rendered:  dict[int, tuple] = {}      # client_id → (el, az, radius)
    _MIN_CHANGE_DEG = 0.5                       # ignore sub-half-degree jitter
    _MIN_INTERVAL   = 0.05                      # seconds between renders

    def render_for_camera(client, cam_handle, force: bool = False):
        """Extract orbit params from camera, render MIP, push to client."""
        el_deg, az_deg, radius = camera_to_orbit(cam_handle)
        beta = beta_slider.value
        cid = client.client_id

        # Skip if camera hasn't moved meaningfully
        if not force and cid in _last_rendered:
            prev = _last_rendered[cid]
            if (abs(el_deg - prev[0]) < _MIN_CHANGE_DEG and
                abs(az_deg - prev[1]) < _MIN_CHANGE_DEG and
                abs(radius - prev[2]) < 0.05):
                return

        # Acquire lock so only one GPU render runs at a time
        if not render_lock.acquire(blocking=False):
            # Store as pending — will be picked up after current render
            _pending_render[cid] = {
                'client': client, 'cam': cam_handle, 'time': time.time()
            }
            return
        try:
            t0 = time.time()
            frame = render_mip_frame(
                gaussians, cam_render, el_deg, az_deg, radius, beta, device
            )
            dt = time.time() - t0
            client.scene.set_background_image(
                frame, format="jpeg", jpeg_quality=85,
            )
            _last_rendered[cid] = (el_deg, az_deg, radius)
            status.content = (
                f"**Epoch {epoch}** | K={K} | {res}x{res} | "
                f"az={az_deg:.0f}° el={el_deg:.0f}° r={radius:.1f} | "
                f"{dt*1000:.0f}ms"
            )
        except Exception as e:
            print(f"  Render error (el={el_deg:.1f} az={az_deg:.1f}): {e}")
        finally:
            render_lock.release()

            # Process any pending render that arrived while we were busy
            if cid in _pending_render:
                pending = _pending_render.pop(cid)
                if time.time() - pending['time'] < 1.0:  # not too stale
                    render_for_camera(pending['client'], pending['cam'])

    @server.on_client_connect
    def on_connect(client: viser.ClientHandle) -> None:
        """Register a camera-update callback for each client."""
        print(f"  Client {client.client_id} connected — attaching camera tracker")

        @client.camera.on_update
        def _on_camera_change(cam_handle) -> None:
            render_for_camera(client, cam_handle)

        # Render initial view for this specific client
        render_for_camera(client, client.camera)

    # ── GUI callbacks ──
    @render_btn.on_click
    def _(_) -> None:
        # Force re-render for ALL connected clients
        for cid, client in server.get_clients().items():
            render_for_camera(client, client.camera, force=True)

    @beta_slider.on_update
    def _(_) -> None:
        for cid, client in server.get_clients().items():
            render_for_camera(client, client.camera, force=True)

    print(f"\n--- Interactive MIP Gaussian Viewer ---")
    print(f"  URL: http://localhost:{args.port}")
    print(f"  Rendering: {res}x{res} MIP projections (server-side, per-camera)")
    print(f"  Epoch {epoch}, K={K} Gaussians, beta={default_beta}")
    print(f"  Drag to orbit — MIP re-renders on every camera update")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()


if __name__ == "__main__":
    main()
