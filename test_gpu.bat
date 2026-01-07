@echo off
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%
set CUPY_ACCELERATORS=cub
python -c "import cupy as cp; print('Compiling kernels for your GPU (first run is slow)...'); a = cp.ones(1000); print('GPU TEST: SUCCESS'); print(f'Sum: {float(cp.sum(a))}')"
