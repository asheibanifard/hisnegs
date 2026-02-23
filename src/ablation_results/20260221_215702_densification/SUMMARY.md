# Ablation Study: densification

**Date**: 2026-02-21 21:57:13

**Total Experiments**: 4

## Results Overview

| Experiment | Success | Notes |
|------------|---------|-------|
| no_densify | ❌ | - |
| with_densify | ❌ | - |
| densify_early_stop | ❌ | - |
| densify_aggressive | ❌ | - |

## Detailed Results

### no_densify

- **Config**: `ablation_results/20260221_215702_densification/configs/no_densify.yml`
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

### with_densify

- **Config**: `ablation_results/20260221_215702_densification/configs/with_densify.yml`
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

### densify_early_stop

- **Config**: `ablation_results/20260221_215702_densification/configs/densify_early_stop.yml`
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

### densify_aggressive

- **Config**: `ablation_results/20260221_215702_densification/configs/densify_aggressive.yml`
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

