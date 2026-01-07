@echo off
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH%
cd /d "C:\Users\batyr\OneDrive\Desktop\Projects\Physics\compute-physics-benchmarks"
python experiments/run_gpu_scaling.py
