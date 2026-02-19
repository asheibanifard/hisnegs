# ---------------------------------------------------------------------------
# Custom autograd function wrapping the CUDA kernels
# ---------------------------------------------------------------------------
import torch
import gaussian_eval_cuda
import torch.nn.functional as F

def _build_L_chol(log_scales: torch.Tensor, quaternions: torch.Tensor, eps: float = 1e-5):
    """
    Compute Cholesky factor from learnable (log_scales, quaternions).
    This function is differentiable through PyTorch autograd.

    Returns L such that  L Lᵀ = R diag(s²) Rᵀ + εI.
    """
    K = log_scales.shape[0]
    scales = torch.exp(log_scales).clamp(1e-5, 1e2)
    q = F.normalize(quaternions, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(K, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    S2 = torch.diag_embed(scales ** 2)
    Sigma = R @ S2 @ R.transpose(-2, -1)
    Sigma_reg = Sigma + eps * torch.eye(3, device=Sigma.device).unsqueeze(0)
    return torch.linalg.cholesky(Sigma_reg.float())


class _GaussianEvalCUDA(torch.autograd.Function):
    """
    Wraps the CUDA gaussian_eval_cuda.forward / .backward kernels.

    The kernels operate on (x, means, L_chol, amplitudes) and produce (N,K).
    The backward kernel returns grad_x, grad_means, grad_amplitudes —
    but NOT grad_L_chol.

    To propagate gradients to log_scales and quaternions we:
      1) Compute grad_L_chol analytically from the Mahalanobis distance
      2) Recompute L_chol from (log_scales, quaternions) inside backward
         with autograd enabled, then call torch.autograd.grad to chain
         grad_L_chol → grad_log_scales, grad_quaternions.
    """

    @staticmethod
    def forward(ctx, x, means, log_scales, quaternions, log_amplitudes, L_chol_detached):
        """
        Args:
            x:                (N, 3) query points
            means:            (K, 3) Gaussian centres
            log_scales:       (K, 3) learnable log-scales
            quaternions:      (K, 4) learnable quaternions
            log_amplitudes:   (K,) learnable log-amplitudes
            L_chol_detached:  (K, 3, 3) precomputed Cholesky factor (detached)
        """
        amplitudes = torch.exp(log_amplitudes.clamp(-10.0, 6.0))

        # CUDA forward: returns (N, K)
        vals_nk = gaussian_eval_cuda.forward(
            x.contiguous().float(),
            means.contiguous().float(),
            L_chol_detached.contiguous().float(),
            amplitudes.detach().contiguous().float(),
        )

        # Sum over K → (N,)
        output = vals_nk.sum(dim=1)

        ctx.save_for_backward(
            x, means, log_scales, quaternions, log_amplitudes,
            L_chol_detached, amplitudes, vals_nk,
        )
        return output.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output_n):
        (
            x, means, log_scales, quaternions, log_amplitudes,
            L_chol, amplitudes, vals_nk,
        ) = ctx.saved_tensors
        N, K = vals_nk.shape

        # Expand (N,) → (N, K) since output = vals_nk.sum(dim=1)
        grad_nk = grad_output_n[:, None].expand(N, K).contiguous()

        # CUDA backward: returns [grad_x, grad_means, grad_L_chol, grad_amplitudes]
        # The kernel already computes grad_L_chol via analytic differentiation
        # of the forward substitution, accumulated over all N points per Gaussian.
        cuda_grads = gaussian_eval_cuda.backward(
            grad_nk.float(),
            x.contiguous().float(),
            means.contiguous().float(),
            L_chol.contiguous().float(),
            amplitudes.detach().contiguous().float(),
            vals_nk.contiguous().float(),
        )
        grad_x = cuda_grads[0]               # (N, 3)
        grad_means = cuda_grads[1]            # (K, 3)
        grad_L_chol = cuda_grads[2]           # (K, 3, 3) — from CUDA kernel
        grad_amplitudes_raw = cuda_grads[3]   # (K,)

        # Ensure only lower-triangular entries are used
        grad_L_chol = torch.tril(grad_L_chol)

        # --- Chain grad_L_chol → grad_log_scales, grad_quaternions ---
        # Recompute L_chol from (log_scales, quaternions) WITH autograd,
        # then use torch.autograd.grad to propagate.
        with torch.enable_grad():
            ls = log_scales.detach().requires_grad_(True)
            qt = quaternions.detach().requires_grad_(True)
            L_recomp = _build_L_chol(ls, qt)
            grads = torch.autograd.grad(
                L_recomp, [ls, qt],
                grad_outputs=grad_L_chol,
                allow_unused=True,
            )
            grad_log_scales = grads[0]
            grad_quaternions = grads[1]

        # grad_log_amplitudes: chain rule through exp(clamp(log_amp))
        grad_log_amplitudes = grad_amplitudes_raw * amplitudes

        # Return grads for: x, means, log_scales, quaternions, log_amplitudes, L_chol
        return (
            grad_x.to(x.dtype),
            grad_means.to(means.dtype),
            grad_log_scales,
            grad_quaternions,
            grad_log_amplitudes.to(log_amplitudes.dtype),
            None,  # L_chol_detached — no grad needed
        )


_gaussian_eval_cuda_fn = _GaussianEvalCUDA.apply


# Gradient supervision uses direct field evaluation with PyTorch autograd
# The field() call will automatically use CUDA kernels when available

class _GradientSupervisionCUDA(torch.autograd.Function):
    """
    Fused CUDA gradient supervision with custom backward.
    
    Forward: uses CUDA kernel to evaluate field at 4 points and compute L1 loss.
    Backward: uses CUDA kernel to compute gradients w.r.t. means, L_chol, amplitudes,
    then chains L_chol gradients to log_scales and quaternions via PyTorch autograd.
    """

    @staticmethod
    def forward(ctx, x_center, x_dx, x_dy, x_dz, v_center, v_dx, v_dy, v_dz,
                means, log_scales, quaternions, log_amplitudes):
        L_chol = _build_L_chol(log_scales, quaternions)
        amps = torch.exp(log_amplitudes.clamp(-10.0, 6.0))

        # CUDA forward: returns [grad_loss (N,), pred_sums (N, 4)]
        results = gaussian_eval_cuda.gradient_supervision(
            x_center.contiguous().float(),
            x_dx.contiguous().float(),
            x_dy.contiguous().float(),
            x_dz.contiguous().float(),
            v_center.contiguous().float(),
            v_dx.contiguous().float(),
            v_dy.contiguous().float(),
            v_dz.contiguous().float(),
            means.contiguous().float(),
            L_chol.detach().contiguous().float(),
            amps.detach().contiguous().float(),
        )
        grad_loss = results[0]   # (N,)
        pred_sums = results[1]   # (N, 4)

        ctx.save_for_backward(
            x_center, x_dx, x_dy, x_dz, v_center, v_dx, v_dy, v_dz,
            means, log_scales, quaternions, log_amplitudes,
            L_chol, amps, pred_sums,
        )
        return grad_loss

    @staticmethod
    def backward(ctx, grad_output):
        (x_center, x_dx, x_dy, x_dz, v_center, v_dx, v_dy, v_dz,
         means, log_scales, quaternions, log_amplitudes,
         L_chol, amps, pred_sums) = ctx.saved_tensors

        # CUDA backward: returns [grad_means, grad_L (K,3,3), grad_amplitudes]
        cuda_grads = gaussian_eval_cuda.gradient_supervision_backward(
            grad_output.contiguous().float(),
            x_center.contiguous().float(),
            x_dx.contiguous().float(),
            x_dy.contiguous().float(),
            x_dz.contiguous().float(),
            v_center.contiguous().float(),
            v_dx.contiguous().float(),
            v_dy.contiguous().float(),
            v_dz.contiguous().float(),
            means.contiguous().float(),
            L_chol.detach().contiguous().float(),
            amps.detach().contiguous().float(),
            pred_sums.contiguous().float(),
        )
        grad_means = cuda_grads[0]           # (K, 3)
        grad_L_chol = torch.tril(cuda_grads[1])  # (K, 3, 3)
        grad_amplitudes_raw = cuda_grads[2]  # (K,)

        # Chain grad_L_chol → grad_log_scales, grad_quaternions
        with torch.enable_grad():
            ls = log_scales.detach().requires_grad_(True)
            qt = quaternions.detach().requires_grad_(True)
            L_recomp = _build_L_chol(ls, qt)
            grads = torch.autograd.grad(
                L_recomp, [ls, qt],
                grad_outputs=grad_L_chol,
                allow_unused=True,
            )
            grad_log_scales = grads[0]
            grad_quaternions = grads[1]

        # Chain grad_amplitudes through exp(clamp(log_amp))
        grad_log_amplitudes = grad_amplitudes_raw * amps

        return (
            None, None, None, None,  # x_center, x_dx, x_dy, x_dz
            None, None, None, None,  # v_center, v_dx, v_dy, v_dz
            grad_means.to(means.dtype),
            grad_log_scales,
            grad_quaternions,
            grad_log_amplitudes.to(log_amplitudes.dtype),
        )


_gradient_supervision_cuda_fn = _GradientSupervisionCUDA.apply


class _AnalyticalFieldGradCUDA(torch.autograd.Function):
    """
    Computes the analytical spatial gradient ∇_x f(x) of the Gaussian field.

    ∇_x f(x) = Σ_k -v_k · L_k^{-T} y_k, where y_k = L_k^{-1}(x - μ_k)

    This is computed alongside f(x) in a single CUDA kernel, but only the
    gradient output (N, 3) participates in the loss. The reconstruction loss
    goes through the existing _GaussianEvalCUDA path separately.
    """

    @staticmethod
    def forward(ctx, x, means, log_scales, quaternions, log_amplitudes, L_chol_detached):
        amplitudes = torch.exp(log_amplitudes.clamp(-10.0, 6.0))

        results = gaussian_eval_cuda.forward_with_field_grad(
            x.contiguous().float(),
            means.contiguous().float(),
            L_chol_detached.contiguous().float(),
            amplitudes.detach().contiguous().float(),
        )
        # results[0] = val (N,), results[1] = field_grad (N, 3)
        field_grad = results[1]

        ctx.save_for_backward(
            x, means, log_scales, quaternions, log_amplitudes,
            L_chol_detached, amplitudes,
        )
        return field_grad

    @staticmethod
    def backward(ctx, grad_field_grad):
        (
            x, means, log_scales, quaternions, log_amplitudes,
            L_chol, amplitudes,
        ) = ctx.saved_tensors

        cuda_grads = gaussian_eval_cuda.analytical_grad_backward(
            grad_field_grad.contiguous().float(),
            x.contiguous().float(),
            means.contiguous().float(),
            L_chol.contiguous().float(),
            amplitudes.detach().contiguous().float(),
        )
        grad_means = cuda_grads[0]                   # (K, 3)
        grad_L_chol = torch.tril(cuda_grads[1])      # (K, 3, 3)
        grad_amplitudes_raw = cuda_grads[2]           # (K,)

        # Chain grad_L_chol → grad_log_scales, grad_quaternions
        with torch.enable_grad():
            ls = log_scales.detach().requires_grad_(True)
            qt = quaternions.detach().requires_grad_(True)
            L_recomp = _build_L_chol(ls, qt)
            grads = torch.autograd.grad(
                L_recomp, [ls, qt],
                grad_outputs=grad_L_chol,
                allow_unused=True,
            )
            grad_log_scales = grads[0]
            grad_quaternions = grads[1]

        # Chain grad_amplitudes through exp(clamp(log_amp))
        grad_log_amplitudes = grad_amplitudes_raw * amplitudes

        return (
            None,  # x
            grad_means.to(means.dtype),
            grad_log_scales,
            grad_quaternions,
            grad_log_amplitudes.to(log_amplitudes.dtype),
            None,  # L_chol_detached
        )


_analytical_field_grad_cuda_fn = _AnalyticalFieldGradCUDA.apply
