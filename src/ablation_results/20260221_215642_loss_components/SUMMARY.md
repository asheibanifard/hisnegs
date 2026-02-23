# Ablation Study: loss_components

**Date**: 2026-02-21 21:57:02

**Total Experiments**: 7

## Results Overview

| Experiment | Success | Notes |
|------------|---------|-------|
| baseline | ❌ | - |
| no_grad_loss | ❌ | - |
| no_tube_reg | ❌ | - |
| no_cross_reg | ❌ | - |
| no_scale_reg | ❌ | - |
| no_regularizers | ❌ | - |
| reconstruction_only | ❌ | - |

## Detailed Results

### baseline

- **Config**: `ablation_results/20260221_215642_loss_components/configs/baseline.yml`
- **Success**: False

**Error Output**:
```
Traceback (most recent call last):
  File "/workspace/hisnegs/src/run.py", line 10, in <module>
    from model import GaussianMixtureField
  File "/workspace/hisnegs/src/model.py", line 7, in <module>
    from cuda_kernels import _build_L_chol, _gaussian_eval_cuda_fn
  File "/workspace/hisnegs/src/cuda_kernels.py", line 5, in <module>
    import gaussian_eval_cuda
ModuleNotFoundError: No module named 'gaussian_eval_cuda'

```

### no_grad_loss

- **Config**: `ablation_results/20260221_215642_loss_components/configs/no_grad_loss.yml`
- **Success**: False

**Error Output**:
```
Traceback (most recent call last):
  File "/workspace/hisnegs/src/run.py", line 10, in <module>
    from model import GaussianMixtureField
  File "/workspace/hisnegs/src/model.py", line 7, in <module>
    from cuda_kernels import _build_L_chol, _gaussian_eval_cuda_fn
  File "/workspace/hisnegs/src/cuda_kernels.py", line 5, in <module>
    import gaussian_eval_cuda
ModuleNotFoundError: No module named 'gaussian_eval_cuda'

```

### no_tube_reg

- **Config**: `ablation_results/20260221_215642_loss_components/configs/no_tube_reg.yml`
- **Success**: False

**Error Output**:
```
Traceback (most recent call last):
  File "/workspace/hisnegs/src/run.py", line 10, in <module>
    from model import GaussianMixtureField
  File "/workspace/hisnegs/src/model.py", line 7, in <module>
    from cuda_kernels import _build_L_chol, _gaussian_eval_cuda_fn
  File "/workspace/hisnegs/src/cuda_kernels.py", line 5, in <module>
    import gaussian_eval_cuda
ModuleNotFoundError: No module named 'gaussian_eval_cuda'

```

### no_cross_reg

- **Config**: `ablation_results/20260221_215642_loss_components/configs/no_cross_reg.yml`
- **Success**: False

**Error Output**:
```
Traceback (most recent call last):
  File "/workspace/hisnegs/src/run.py", line 10, in <module>
    from model import GaussianMixtureField
  File "/workspace/hisnegs/src/model.py", line 7, in <module>
    from cuda_kernels import _build_L_chol, _gaussian_eval_cuda_fn
  File "/workspace/hisnegs/src/cuda_kernels.py", line 5, in <module>
    import gaussian_eval_cuda
ModuleNotFoundError: No module named 'gaussian_eval_cuda'

```

### no_scale_reg

- **Config**: `ablation_results/20260221_215642_loss_components/configs/no_scale_reg.yml`
- **Success**: False

**Error Output**:
```
Traceback (most recent call last):
  File "/workspace/hisnegs/src/run.py", line 10, in <module>
    from model import GaussianMixtureField
  File "/workspace/hisnegs/src/model.py", line 7, in <module>
    from cuda_kernels import _build_L_chol, _gaussian_eval_cuda_fn
  File "/workspace/hisnegs/src/cuda_kernels.py", line 5, in <module>
    import gaussian_eval_cuda
ModuleNotFoundError: No module named 'gaussian_eval_cuda'

```

### no_regularizers

- **Config**: `ablation_results/20260221_215642_loss_components/configs/no_regularizers.yml`
- **Success**: False

**Error Output**:
```
Traceback (most recent call last):
  File "/workspace/hisnegs/src/run.py", line 10, in <module>
    from model import GaussianMixtureField
  File "/workspace/hisnegs/src/model.py", line 7, in <module>
    from cuda_kernels import _build_L_chol, _gaussian_eval_cuda_fn
  File "/workspace/hisnegs/src/cuda_kernels.py", line 5, in <module>
    import gaussian_eval_cuda
ModuleNotFoundError: No module named 'gaussian_eval_cuda'

```

### reconstruction_only

- **Config**: `ablation_results/20260221_215642_loss_components/configs/reconstruction_only.yml`
- **Success**: False

**Error Output**:
```
Traceback (most recent call last):
  File "/workspace/hisnegs/src/run.py", line 10, in <module>
    from model import GaussianMixtureField
  File "/workspace/hisnegs/src/model.py", line 7, in <module>
    from cuda_kernels import _build_L_chol, _gaussian_eval_cuda_fn
  File "/workspace/hisnegs/src/cuda_kernels.py", line 5, in <module>
    import gaussian_eval_cuda
ModuleNotFoundError: No module named 'gaussian_eval_cuda'

```

