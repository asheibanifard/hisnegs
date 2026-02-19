import io
import math
import os
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/venv/hisnegs")
os.environ.setdefault("CPATH", f"/venv/hisnegs/targets/x86_64-linux/include:{os.environ.get('CPATH', '')}")
os.environ.setdefault("LIBRARY_PATH", f"/venv/hisnegs/targets/x86_64-linux/lib:{os.environ.get('LIBRARY_PATH', '')}")
os.environ.setdefault(
    "LD_LIBRARY_PATH",
    "/venv/hisnegs/lib:/venv/hisnegs/lib/python3.10/site-packages/torch/lib:/venv/hisnegs/targets/x86_64-linux/lib:"
    f"{os.environ.get('LD_LIBRARY_PATH', '')}",
)
os.environ.setdefault("CC", "/venv/hisnegs/bin/x86_64-conda-linux-gnu-gcc")
os.environ.setdefault("CXX", "/venv/hisnegs/bin/x86_64-conda-linux-gnu-g++")

import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image
import gsplat
import tifffile

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = APP_DIR.parent / "checkpoints" / "render_skpt" / "splat_step10000.pt"


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (torch.norm(v) + eps)


def orbit_viewmat(yaw_deg: float, pitch_deg: float, radius: float, device: torch.device) -> torch.Tensor:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    eye = torch.tensor(
        [
            radius * math.cos(pitch) * math.sin(yaw),
            radius * math.sin(pitch),
            radius * math.cos(pitch) * math.cos(yaw),
        ],
        dtype=torch.float32,
        device=device,
    )

    target = torch.zeros(3, dtype=torch.float32, device=device)
    forward = normalize(target - eye)
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)

    right = torch.linalg.cross(forward, world_up)
    if right.norm() < 1e-6:
        world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        right = torch.linalg.cross(forward, world_up)
    right = normalize(right)
    up = normalize(torch.linalg.cross(right, forward))

    R = torch.stack([right, -up, forward], dim=0)
    t = -R @ eye

    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    return viewmat


def intrinsics(width: int, height: int, fov_x_deg: float, device: torch.device) -> torch.Tensor:
    fx = width / (2.0 * math.tan(math.radians(fov_x_deg) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    return K


def quat_scale_to_covars(quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    q = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    K = quats.shape[0]
    R = torch.zeros(K, 3, 3, device=quats.device, dtype=quats.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    S2 = torch.diag_embed(scales * scales)
    return R @ S2 @ R.transpose(-2, -1)


def compute_aspect_scales_from_shape(vol_shape_zyx: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    z, y, x = vol_shape_zyx
    max_dim = float(max(x, y, z))
    return torch.tensor([x / max_dim, y / max_dim, z / max_dim], dtype=torch.float32, device=device)


class GSplatDemo:
    def __init__(self, checkpoint_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self._load_checkpoint()
        self._warmup()

    def _load_checkpoint(self):
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.means = ckpt["means"].float().to(self.device)
        self.quats = ckpt["quaternions"].float().to(self.device)
        self.quats = self.quats / self.quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.scales = torch.exp(ckpt["log_scales"].float().to(self.device)).clamp(1e-5, 1e2)
        self.opacities = torch.exp(ckpt["log_amplitudes"].float().to(self.device)).clamp(0.0, 1.0)

        op_norm = self.opacities / self.opacities.max().clamp(min=1e-6)
        self.colors = op_norm.pow(0.35).unsqueeze(-1).repeat(1, 3)

        shape_env = os.environ.get("VOL_SHAPE", "").strip()
        if shape_env:
            try:
                z, y, x = [int(v.strip()) for v in shape_env.split(",")]
                self.aspect_scales = compute_aspect_scales_from_shape((z, y, x), self.device)
                print(f"Using VOL_SHAPE aspect scales: {self.aspect_scales.tolist()}")
                return
            except Exception:
                pass

        default_tif = APP_DIR.parent.parent / "dataset" / "10-2900-control-cell-05_cropped_corrected.tif"
        tif_path = Path(os.environ.get("VOL_TIF", str(default_tif)))
        if tif_path.exists():
            vol = tifffile.imread(str(tif_path))
            self.aspect_scales = compute_aspect_scales_from_shape(tuple(vol.shape), self.device)
            print(f"Using volume aspect scales from {tif_path}: {self.aspect_scales.tolist()}")
        else:
            self.aspect_scales = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)
            print("WARNING: volume shape not found; using isotropic aspect scales [1,1,1]")

    def _warmup(self):
        try:
            self.render(yaw=0.0, pitch=0.0, radius=3.5, width=64, height=64, fov=50.0, max_gaussians=1024)
        except Exception as exc:
            print(f"Warmup warning: {exc}")

    @torch.inference_mode()
    def render(
        self,
        yaw: float,
        pitch: float,
        radius: float,
        width: int,
        height: int,
        fov: float,
        max_gaussians: int,
    ) -> np.ndarray:
        means = self.means
        quats = self.quats
        scales = self.scales
        opacities = self.opacities
        colors = self.colors

        if max_gaussians > 0 and means.shape[0] > max_gaussians:
            idx = torch.topk(opacities, k=max_gaussians, largest=True).indices
            means = means[idx]
            quats = quats[idx]
            scales = scales[idx]
            opacities = opacities[idx]
            colors = colors[idx]

        scales = scales.clamp(min=1e-5, max=1e2)
        opacities = opacities.clamp(min=0.0, max=1.0)

        s = self.aspect_scales.to(means.device)
        means = means * s.unsqueeze(0)
        covars = quat_scale_to_covars(quats, scales)
        S = torch.diag(s)
        covars = S.unsqueeze(0) @ covars @ S.unsqueeze(0).transpose(-2, -1)

        viewmats = orbit_viewmat(yaw, pitch, radius, self.device).unsqueeze(0)
        Ks = intrinsics(width, height, fov, self.device).unsqueeze(0)

        rgb, alpha, _ = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=0.01,
            far_plane=100.0,
            radius_clip=0.5,
            eps2d=0.5,
            render_mode="RGB",
            rasterize_mode="antialiased",
            packed=True,
            covars=covars,
        )

        img = rgb[0].clamp(min=0.0)
        a = alpha[0].clamp(0.0, 1.0)
        bg = torch.full_like(img, 0.02)
        img = img * a + bg * (1.0 - a)
        exposure = float(os.environ.get("RENDER_EXPOSURE", "1.0"))
        img = (img * exposure).clamp(0.0, 1.0)
        img = img.pow(1.0 / 2.2)
        img = img.detach().cpu().numpy()
        return (img * 255.0).astype(np.uint8)


app = Flask(__name__, template_folder="templates")

ckpt_env = os.environ.get("SPLAT_CKPT", str(DEFAULT_CHECKPOINT))
demo = GSplatDemo(Path(ckpt_env))


@app.get("/")
def index():
    return render_template("index.html", checkpoint=str(demo.checkpoint_path), device=str(demo.device), n_gaussians=int(demo.means.shape[0]))


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True, "device": str(demo.device), "checkpoint": str(demo.checkpoint_path), "gaussians": int(demo.means.shape[0])})


@app.get("/render")
def render_endpoint():
    yaw = float(request.args.get("yaw", 0.0))
    pitch = float(request.args.get("pitch", 0.0))
    radius = float(request.args.get("radius", 3.5))
    width = int(request.args.get("w", 640))
    height = int(request.args.get("h", 640))
    fov = float(request.args.get("fov", 50.0))
    max_gaussians = int(request.args.get("max_k", 0))

    width = max(64, min(width, 1280))
    height = max(64, min(height, 1280))
    pitch = max(-89.0, min(pitch, 89.0))
    radius = max(0.5, min(radius, 20.0))

    image = demo.render(yaw, pitch, radius, width, height, fov, max_gaussians)
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
