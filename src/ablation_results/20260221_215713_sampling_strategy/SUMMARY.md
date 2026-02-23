# Ablation Study: sampling_strategy

**Date**: 2026-02-21 21:57:24

**Total Experiments**: 4

## Results Overview

| Experiment | Success | Notes |
|------------|---------|-------|
| uniform_sampling | ❌ | - |
| intensity_weighted | ❌ | - |
| more_volume_samples | ❌ | - |
| fewer_volume_samples | ❌ | - |

## Detailed Results

### uniform_sampling

- **Config**: `ablation_results/20260221_215713_sampling_strategy/configs/uniform_sampling.yml`
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

### intensity_weighted

- **Config**: `ablation_results/20260221_215713_sampling_strategy/configs/intensity_weighted.yml`
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

### more_volume_samples

- **Config**: `ablation_results/20260221_215713_sampling_strategy/configs/more_volume_samples.yml`
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

### fewer_volume_samples

- **Config**: `ablation_results/20260221_215713_sampling_strategy/configs/fewer_volume_samples.yml`
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

