# Ablation Study: initialization

**Date**: 2026-02-21 21:57:41

**Total Experiments**: 6

## Results Overview

| Experiment | Success | Notes |
|------------|---------|-------|
| init_swc | ❌ | - |
| init_random | ❌ | - |
| init_scale_small | ❌ | - |
| init_scale_large | ❌ | - |
| fewer_gaussians | ❌ | - |
| more_gaussians | ❌ | - |

## Detailed Results

### init_swc

- **Config**: `ablation_results/20260221_215724_initialization/configs/init_swc.yml`
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

### init_random

- **Config**: `ablation_results/20260221_215724_initialization/configs/init_random.yml`
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

### init_scale_small

- **Config**: `ablation_results/20260221_215724_initialization/configs/init_scale_small.yml`
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

### init_scale_large

- **Config**: `ablation_results/20260221_215724_initialization/configs/init_scale_large.yml`
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

### fewer_gaussians

- **Config**: `ablation_results/20260221_215724_initialization/configs/fewer_gaussians.yml`
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

### more_gaussians

- **Config**: `ablation_results/20260221_215724_initialization/configs/more_gaussians.yml`
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

