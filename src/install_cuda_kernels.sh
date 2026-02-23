#!/bin/bash
# Comprehensive CUDA kernel installation script
# Builds all CUDA extensions for the project

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         CUDA Kernel Installation for NeuroGS              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Change to src directory
cd "$(dirname "$0")"

# Check Python and PyTorch
echo "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi
echo "✓ Python found: $(which python3)"

# Check PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    echo "❌ Error: PyTorch not installed"
    echo "   Install with: pip install torch"
    exit 1
fi
echo "✓ PyTorch installed"

# Check CUDA availability
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo "⚠ Warning: CUDA not available in PyTorch"
    echo "   Training will use CPU (much slower)"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ CUDA available"
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    echo "  CUDA version: $CUDA_VERSION"
fi

# Check for CUDA compiler
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA compiler found: $(which nvcc)"
    NVCC_VERSION=$(nvcc --version | grep release | awk '{print $6}' | cut -c2-)
    echo "  nvcc version: $NVCC_VERSION"
else
    echo "⚠ Warning: nvcc not found"
    echo "   CUDA extensions will not be compiled"
    echo "   Code will fallback to PyTorch (slower but works)"
    echo ""
fi

echo ""
echo "Building CUDA extensions..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Track success/failure
MAIN_SUCCESS=false
SPLAT_SUCCESS=false
SPLAT_MIP_SUCCESS=false

# 1. Build main Gaussian evaluation kernel
echo "[1/3] Building gaussian_eval_cuda..."
if [ -f "gaussian_eval_cuda.cu" ] && [ -f "setup_cuda.py" ]; then
    if python3 setup_cuda.py build_ext --inplace 2>&1 | tee /tmp/build_main.log; then
        echo "✓ gaussian_eval_cuda built successfully"
        MAIN_SUCCESS=true
    else
        echo "✗ gaussian_eval_cuda build failed (will use PyTorch fallback)"
        echo "  Check /tmp/build_main.log for details"
    fi
else
    echo "⚠ Skipping - files not found"
fi
echo ""

# 2. Build standard splatting kernel
echo "[2/3] Building splat_cuda..."
cd renderer
if [ -f "splat_cuda.cu" ] && [ -f "setup_splat_cuda.py" ]; then
    if python3 setup_splat_cuda.py build_ext --inplace 2>&1 | tee /tmp/build_splat.log; then
        echo "✓ splat_cuda built successfully"
        SPLAT_SUCCESS=true
    else
        echo "✗ splat_cuda build failed (will use fallback)"
        echo "  Check /tmp/build_splat.log for details"
    fi
else
    echo "⚠ Skipping - files not found"
fi
echo ""

# 3. Build MIP splatting kernel
echo "[3/3] Building splat_mip_cuda..."
if [ -f "splat_mip_cuda.cu" ]; then
    # Check if there's a separate setup file or if it uses the same
    if [ -f "setup_splat_mip_cuda.py" ]; then
        if python3 setup_splat_mip_cuda.py build_ext --inplace 2>&1 | tee /tmp/build_mip.log; then
            echo "✓ splat_mip_cuda built successfully"
            SPLAT_MIP_SUCCESS=true
        else
            echo "✗ splat_mip_cuda build failed (will use fallback)"
            echo "  Check /tmp/build_mip.log for details"
        fi
    else
        echo "⚠ Setup file not found - may be included in setup_splat_cuda.py"
        SPLAT_MIP_SUCCESS="unknown"
    fi
else
    echo "⚠ Skipping - files not found"
fi

cd ..

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Installation Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "CUDA Extensions:"
if [ "$MAIN_SUCCESS" = true ]; then
    echo "  ✓ gaussian_eval_cuda    [COMPILED]"
else
    echo "  ✗ gaussian_eval_cuda    [FALLBACK]"
fi

if [ "$SPLAT_SUCCESS" = true ]; then
    echo "  ✓ splat_cuda            [COMPILED]"
else
    echo "  ✗ splat_cuda            [FALLBACK]"
fi

if [ "$SPLAT_MIP_SUCCESS" = true ]; then
    echo "  ✓ splat_mip_cuda        [COMPILED]"
elif [ "$SPLAT_MIP_SUCCESS" = "unknown" ]; then
    echo "  ? splat_mip_cuda        [CHECK LOG]"
else
    echo "  ✗ splat_mip_cuda        [FALLBACK]"
fi

echo ""

if [ "$MAIN_SUCCESS" = true ] || [ "$SPLAT_SUCCESS" = true ] || [ "$SPLAT_MIP_SUCCESS" = true ]; then
    echo "✓ Installation complete with at least one CUDA extension compiled!"
    echo ""
    echo "Performance: CUDA extensions provide 2-3x speedup for training"
else
    echo "⚠ No CUDA extensions were compiled"
    echo ""
    echo "The code will work using PyTorch fallback (slower but functional)"
fi

echo ""
echo "To verify installation, run:"
echo "  python3 -c 'from model import GaussianMixtureField; print(\"Model imported successfully\")'"
echo ""
echo "To start training:"
echo "  python3 run.py --config config.yml"
echo ""
