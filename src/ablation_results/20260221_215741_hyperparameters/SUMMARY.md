# Ablation Study: hyperparameters

**Date**: 2026-02-21 21:57:54

**Total Experiments**: 5

## Results Overview

| Experiment | Success | Notes |
|------------|---------|-------|
| lr_low | ❌ | - |
| lr_high | ❌ | - |
| no_early_stopping | ❌ | - |
| no_mixed_precision | ❌ | - |
| no_grad_clip | ❌ | - |

## Detailed Results

### lr_low

- **Config**: `ablation_results/20260221_215741_hyperparameters/configs/lr_low.yml`
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

### lr_high

- **Config**: `ablation_results/20260221_215741_hyperparameters/configs/lr_high.yml`
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

### no_early_stopping

- **Config**: `ablation_results/20260221_215741_hyperparameters/configs/no_early_stopping.yml`
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

### no_mixed_precision

- **Config**: `ablation_results/20260221_215741_hyperparameters/configs/no_mixed_precision.yml`
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

### no_grad_clip

- **Config**: `ablation_results/20260221_215741_hyperparameters/configs/no_grad_clip.yml`
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

