# MIP Gaussian Splatting Experiments

This folder contains the LaTeX report and all generated figures from experimental evaluations.

## Contents

### Main Report
- **mip_splatting_report.tex** - Complete technical report documenting the MIP Gaussian Splatting pipeline

### Figures from splat_experiments.ipynb
- `fig_performance_benchmark.pdf/png` - Rendering performance vs. resolution comparison
- `fig_visual_quality.pdf/png` - Visual quality assessment across six viewpoints  
- `fig_scalability.pdf/png` - Rendering time vs. number of Gaussians
- `fig_orbit_timing.pdf/png` - 360° orbit rendering timing distribution
- `fig_orbit_strip.pdf/png` - Visual samples from orbit rendering

### Figures from ablation study (mip_splatting_ablation_study.ipynb)
- `ablation_training_curves.pdf/png` - Training curves comparing loss configurations
- `ablation_metric_distributions.pdf/png` - PSNR/SSIM/MAE distributions
- `ablation_efficiency.pdf/png` - Model efficiency (PSNR vs. Gaussian count)
- `ablation_visual_comparison.pdf/png` - Side-by-side visual quality comparison

## Compiling the Report

### Requirements
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

### Compilation
```bash
cd /workspace/hisnegs/experiments
pdflatex mip_splatting_report.tex
pdflatex mip_splatting_report.tex  # Run twice for TOC and references
```

The compiled PDF will be generated as `mip_splatting_report.pdf`.

## Generating Figures

### Option 1: Run splat_experiments.ipynb
Execute all cells in `/workspace/hisnegs/src/renderer/splat_experiments.ipynb` to generate:
- Performance benchmarks
- Visual quality assessments
- Scalability analysis
- Orbit rendering results

Figures will be saved automatically to this directory.

### Option 2: Run ablation study notebook
Execute all cells in `/workspace/hisnegs/mip_splatting_ablation_study.ipynb` to generate:
- Loss component ablation results
- Training dynamics comparison
- Statistical significance tests
- Visual quality comparisons

Figures will be saved automatically to this directory.

## Directory Structure
```
experiments/
├── README.md                          # This file
├── mip_splatting_report.tex          # Main LaTeX report
├── fig_performance_benchmark.pdf     # Performance benchmark
├── fig_visual_quality.pdf            # Visual quality assessment
├── fig_scalability.pdf               # Scalability analysis
├── fig_orbit_timing.pdf              # Orbit timing
├── fig_orbit_strip.pdf               # Orbit visual samples
├── ablation_training_curves.pdf      # Ablation training curves
├── ablation_metric_distributions.pdf # Ablation metrics
├── ablation_efficiency.pdf           # Ablation efficiency
└── ablation_visual_comparison.pdf    # Ablation visual comparison
```

## Figure References in Report

All figures are referenced in the report using their respective labels:
- `\ref{fig:performance}` - Performance benchmark
- `\ref{fig:visual_quality}` - Visual quality
- `\ref{fig:scalability}` - Scalability
- `\ref{fig:orbit_timing}` - Orbit timing
- `\ref{fig:orbit_strip}` - Orbit visuals
- `\ref{fig:ablation}` - Ablation training/metrics
- `\ref{fig:ablation_efficiency}` - Ablation efficiency
- `\ref{fig:ablation_visual}` - Ablation visual comparison

## Notes

- All figures are generated in both PDF (for LaTeX) and PNG (for notebooks) formats
- PDF versions are used in the LaTeX report for vector graphics quality
- Figures are generated at 300 DPI for publication quality
- The report uses relative paths to figures in the same directory
