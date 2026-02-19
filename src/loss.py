
import torch
import torch.nn.functional as F
import gaussian_eval_cuda
from model import GaussianMixtureField
from regularisers import tubular_regulariser, cross_section_symmetry_reg
from cuda_kernels import (
    _gradient_supervision_cuda_fn,
    _analytical_field_grad_cuda_fn,
    _build_L_chol,
)

HAS_CUDA_EXTENSION = True

def loss_volume(
    field: GaussianMixtureField,
    x: torch.Tensor,
    v: torch.Tensor,
    # neighbour data (optional — pass None to skip gradient loss)
    x_dx: torch.Tensor | None = None,
    v_dx: torch.Tensor | None = None,
    x_dy: torch.Tensor | None = None,
    v_dy: torch.Tensor | None = None,
    x_dz: torch.Tensor | None = None,
    v_dz: torch.Tensor | None = None,
    *,
    w_grad: float = 0.3,
    w_tube: float = 1e-4,
    w_cross: float = 1e-4,
    w_scale: float = 5e-4,
    scale_target: float | None = 0.03,
) -> tuple[torch.Tensor, dict]:

    pred = field(x)
    l_rec = F.mse_loss(pred, v)

    # --- gradient supervision (finite differences) ---
    l_grad = torch.zeros((), device=x.device)
    if x_dx is not None and w_grad > 0:
        use_cuda_grad = (
            HAS_CUDA_EXTENSION
            and x.is_cuda
            and hasattr(gaussian_eval_cuda, 'gradient_supervision_backward')
        )

        use_analytical_grad = (
            HAS_CUDA_EXTENSION
            and x.is_cuda
            and hasattr(gaussian_eval_cuda, 'analytical_grad_backward')
        )

        if use_analytical_grad:
            # Analytical gradient: compute ∇_x f(x) in single kernel call
            L_chol = _build_L_chol(field.log_scales, field.quaternions).detach()
            field_grad = _analytical_field_grad_cuda_fn(
                x, field.means, field.log_scales, field.quaternions,
                field.log_amplitudes, L_chol,
            )  # (N, 3)

            # Ground truth finite differences from volume
            # field_grad is the derivative w.r.t. normalised coords;
            # multiply by the step in normalised coords to predict the difference
            deltas = torch.stack([
                x_dx[:, 0] - x[:, 0],
                x_dy[:, 1] - x[:, 1],
                x_dz[:, 2] - x[:, 2],
            ], dim=-1)  # (N, 3)

            pred_diff = field_grad * deltas  # predicted finite differences (N, 3)
            gt_diff = torch.stack([v_dx - v, v_dy - v, v_dz - v], dim=-1)  # (N, 3)
            l_grad = F.l1_loss(pred_diff, gt_diff)
        elif use_cuda_grad:
            # Fused CUDA kernel with custom backward — fastest path
            grad_loss_per_point = _gradient_supervision_cuda_fn(
                x, x_dx, x_dy, x_dz,
                v, v_dx, v_dy, v_dz,
                field.means, field.log_scales, field.quaternions, field.log_amplitudes
            )
            l_grad = grad_loss_per_point.mean()
        else:
            # PyTorch fallback: match SIGNED gradients (preserves edge direction)
            p_dx = field(x_dx)
            p_dy = field(x_dy)
            p_dz = field(x_dz)
            l_grad = (
                F.l1_loss(p_dx - pred, v_dx - v)  # signed gradient in x
                + F.l1_loss(p_dy - pred, v_dy - v)  # signed gradient in y
                + F.l1_loss(p_dz - pred, v_dz - v)  # signed gradient in z
            )

    # --- covariance regularisers ---
    Sigma = field.get_covariance_matrices()
    l_tube = tubular_regulariser(Sigma)
    l_csym = cross_section_symmetry_reg(Sigma)

    # --- scale regulariser (prevent blobs) ---
    scales = torch.exp(field.log_scales).clamp(1e-6, 1e2)
    if scale_target is not None:
        l_scale = F.relu(scales - scale_target).mean()
    else:
        l_scale = scales.mean()

    total = (
        l_rec
        + w_grad * l_grad
        + w_tube * l_tube
        + w_cross * l_csym
        + w_scale * l_scale
    )
    parts = {
        "rec": l_rec,
        "grad": l_grad,
        "tube": l_tube,
        "csym": l_csym,
        "scale": l_scale,
    }
    return total, parts

def render_soft_mip_z(
    field: GaussianMixtureField,
    xy: torch.Tensor,
    n_z: int,
    tau: float,
    pt_chunk: int = 16384,
) -> torch.Tensor:
    """Soft z-MIP via LogSumExp along z-rays."""
    device = xy.device
    P = xy.shape[0]
    z_vals = torch.linspace(-1, 1, n_z, device=device, dtype=xy.dtype)

    pts = torch.cat(
        [
            xy[:, None, :].expand(P, n_z, 2),
            z_vals[None, :, None].expand(P, n_z, 1),
        ],
        dim=-1,
    ).reshape(-1, 3)  # (P*n_z, 3)

    vals = []
    for i in range(0, pts.shape[0], pt_chunk):
        vals.append(field(pts[i : i + pt_chunk]))
    v = torch.cat(vals).reshape(P, n_z)

    tau_safe = max(tau, 1e-6)
    return tau_safe * torch.logsumexp(v / tau_safe, dim=1)


def loss_mip(
    field: GaussianMixtureField,
    xy: torch.Tensor,
    mip_gt: torch.Tensor,
    n_z: int,
    tau: float,
    *,
    w_tube: float = 1e-4,
    w_cross: float = 1e-4,
    mip_batch: int = 512,
) -> tuple[torch.Tensor, dict]:
    P = xy.shape[0]
    if P <= mip_batch:
        pred = render_soft_mip_z(field, xy, n_z, tau)
    else:
        chunks = []
        for i in range(0, P, mip_batch):
            chunks.append(render_soft_mip_z(field, xy[i : i + mip_batch], n_z, tau))
        pred = torch.cat(chunks)

    l_img = F.l1_loss(pred, mip_gt)

    Sigma = field.get_covariance_matrices()
    l_tube = tubular_regulariser(Sigma)
    l_csym = cross_section_symmetry_reg(Sigma)

    total = l_img + w_tube * l_tube + w_cross * l_csym
    return total, {"mip": l_img, "tube": l_tube, "csym": l_csym}
