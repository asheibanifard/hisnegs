#!/usr/bin/env python3
"""
Check CUDA extension status and build if needed
"""
import os
import sys
import subprocess

def check_cuda_extension(name):
    """Check if a CUDA extension is available"""
    try:
        __import__(name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 80)
    print("CUDA Extension Status Check")
    print("=" * 80)
    print()
    
    # Change to source directory
    src_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(src_dir)
    print(f"Working directory: {src_dir}")
    print()
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch not installed")
        return 1
    print()
    
    # Check CUDA extensions
    extensions = {
        'gaussian_eval_cuda': 'Gaussian evaluation kernel',
    }
    
    print("CUDA Extension Status:")
    print("-" * 80)
    
    all_available = True
    for ext_name, description in extensions.items():
        available = check_cuda_extension(ext_name)
        status = "✓ AVAILABLE" if available else "✗ NOT FOUND"
        print(f"  {ext_name:25s} {status:15s} ({description})")
        if not available:
            all_available = False
    
    print()
    
    if all_available:
        print("✓ All CUDA extensions are available!")
        print()
        print("You can start training:")
        print("  python3 run.py --config config.yml")
        return 0
    else:
        print("⚠ Some CUDA extensions are not available")
        print()
        print("To build them:")
        print("  1. Make sure CUDA toolkit is installed (nvcc)")
        print("  2. Run: ./install_cuda_kernels.sh")
        print("  3. Or manually: python3 setup_cuda.py build_ext --inplace")
        print()
        print("The code will work with PyTorch fallback (slower but functional)")
        print()
        
        # Try to build automatically
        response = input("Would you like to try building now? (y/N): ")
        if response.lower() == 'y':
            print()
            print("Attempting to build CUDA extensions...")
            print("-" * 80)
            
            try:
                result = subprocess.run(
                    ['python3', 'setup_cuda.py', 'build_ext', '--inplace'],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print("✓ Build successful!")
                    print()
                    # Recheckfor available import gaussian_eval_cuda
                    if check_cuda_extension('gaussian_eval_cuda'):
                        print("✓ gaussian_eval_cuda is now available!")
                        return 0
                    else:
                        print("⚠ Build completed but extension not importable")
                        print("  You may need to restart Python")
                        return 1
                else:
                    print("✗ Build failed")
                    print()
                    print("Error output:")
                    print(result.stderr)
                    return 1
                    
            except Exception as e:
                print(f"✗ Build failed: {e}")
                return 1
        else:
            return 1

if __name__ == "__main__":
    sys.exit(main())
