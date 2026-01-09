# CuPy Setup for RTX 5090 / RTX 50-Series (Blackwell Architecture)

**The Problem**: RTX 5090, 5080, 5070 (Blackwell GPUs) don't work with pre-built CuPy packages because they require CUDA 13+ and compute capability 12.0 (Blackwell), which aren't included in standard CuPy wheels.

**The Solution**: Build CuPy from source with CUDA 13.1.

**Tested on**: RTX 5090 + Windows 11 + Python 3.11 + CUDA 13.1

---

## Prerequisites

- Windows 10/11
- Python 3.9+ installed
- NVIDIA RTX 50-series GPU
- ~5GB disk space
- ~30 minutes

---

## Step 1: Install CUDA Toolkit 13.1

1. Go to: https://developer.nvidia.com/cuda-downloads
2. Select: **Windows → x86_64 → 11 → exe (local)**
3. Download and run the installer
4. Choose **Custom Install**
5. Select:
   - CUDA → Runtime
   - CUDA → Development
   - CUDA → Documentation (optional)
6. **Uncheck** Display Driver (you already have a newer one)
7. Complete installation

**Verify installation:**
```cmd
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
```
You should see `v13.1` listed.

---

## Step 2: Build CuPy from Source

Create a file called `build_cupy_blackwell.bat`:

```batch
@echo off
echo ============================================================
echo CuPy Build for RTX 50-Series (Blackwell) - CUDA 13.1
echo ============================================================
echo.
echo This will take 15-30 minutes.
echo.

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH%

echo [1/4] Checking CUDA 13.1...
nvcc --version
if errorlevel 1 (
    echo ERROR: nvcc not found. Make sure CUDA 13.1 is installed.
    pause
    exit /b 1
)
echo.

echo [2/4] Removing old CuPy installations...
pip uninstall cupy cupy-cuda11x cupy-cuda12x -y 2>nul
echo.

echo [3/4] Installing build dependencies...
pip install cython numpy fastrlock
echo.

echo [4/4] Building CuPy from source (this is the slow part)...
pip install cupy --no-binary cupy

echo.
echo ============================================================
echo Build complete!
echo ============================================================
pause
```

Run the batch file and wait 15-30 minutes.

---

## Step 3: Set Up Environment Variables

**Important**: CUDA 13.1 on Windows puts DLLs in `bin\x64\`, not just `bin\`.

### Option A: Permanent Setup (Recommended)

1. Open **System Properties** → **Environment Variables**
2. Under **User Variables**, edit `Path`
3. Add these two entries:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin
   ```
4. Add new User Variable:
   - Name: `CUDA_PATH`
   - Value: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`
5. Click OK and restart your terminal

### Option B: Per-Session Setup

Create a batch file to set environment before running Python:

```batch
@echo off
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH%
python %*
```

Save as `python_gpu.bat` and use it instead of `python`.

---

## Step 4: Verify Installation

Create `test_cupy.py`:

```python
import cupy as cp

print("Testing CuPy with RTX 50-series...")

# Basic test
a = cp.ones(1000)
print(f"Sum test: {float(cp.sum(a))}")

# Memory info
mem_info = cp.cuda.Device(0).mem_info
print(f"GPU Memory: {mem_info[1] / 1e9:.1f} GB total, {mem_info[0] / 1e9:.1f} GB free")

# Performance test
import time
size = 10000
a = cp.random.randn(size, size)
start = time.time()
b = cp.matmul(a, a)
cp.cuda.Stream.null.synchronize()
elapsed = time.time() - start
print(f"10000x10000 matrix multiply: {elapsed*1000:.1f} ms")

print("\nSUCCESS! Your RTX 50-series is working with CuPy!")
```

Run:
```cmd
python test_cupy.py
```

Expected output:
```
Testing CuPy with RTX 50-series...
Sum test: 1000.0
GPU Memory: 34.2 GB total, 31.5 GB free
10000x10000 matrix multiply: 45.2 ms

SUCCESS! Your RTX 50-series is working with CuPy!
```

---

## Troubleshooting

### Error: `nvrtc-builtins64_131.dll not found`

Your PATH doesn't include the CUDA DLLs. Make sure you added:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64
```
(Note: `bin\x64`, not just `bin`)

### Error: `CUDA_ERROR_NO_BINARY_FOR_GPU`

You're using pre-built CuPy instead of source-built. Run:
```cmd
pip uninstall cupy cupy-cuda11x cupy-cuda12x -y
pip install cupy --no-binary cupy
```

### Error: `nvcc not found` during build

CUDA Toolkit not installed or not in PATH. Verify:
```cmd
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" --version
```

### Build fails with compiler errors

Make sure you have Visual Studio Build Tools installed:
```cmd
pip install setuptools wheel
```

---

## Why This Works

1. **RTX 50-series (Blackwell)** has compute capability 12.0
2. **Pre-built CuPy wheels** only support up to compute capability ~9.x (Ada Lovelace)
3. **CUDA 13.1** is the first CUDA version with full Blackwell support
4. **Building from source** compiles CUDA kernels specifically for your GPU

When you build from source with CUDA 13.1, CuPy compiles PTX code that can be JIT-compiled for your specific GPU architecture at runtime.

---

## Tested Configurations

| GPU | CUDA | CuPy | Status |
|-----|------|------|--------|
| RTX 5090 | 13.1 | 13.6.0 (source) | ✅ Working |

---

## Performance Results (RTX 5090)

Batch integration benchmark (1000 timesteps, dt=0.01):

| Batch Size | CPU Time | GPU Time | Speedup |
|------------|----------|----------|---------|
| 1,000 | 26ms | 295ms | 0.09x |
| 10,000 | 100ms | 311ms | 0.32x |
| 100,000 | 4.3s | 490ms | **8.85x** |
| 1,000,000 | 49.4s | 2.4s | **20.94x** |

GPU crossover point: ~50,000 particles

---

## Credits

Setup guide created by Batyr (ASU Physics)
January 2026

If this helped you, consider starring the repo!

---

## Related: PyTorch for RTX 50-series

Similar issues exist with PyTorch. Check for nightly builds:
```cmd
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu131
```

(Note: URL may change as PyTorch releases official CUDA 13 support)
