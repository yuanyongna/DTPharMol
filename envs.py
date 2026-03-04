"""
2080Ti:
172.30.102.72:22
yzx: 011529yang
root: hadoop421

nvcc --version:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_19:24:38_PDT_2019
Cuda compilation tools, release 10.2, V10.2.89
"""

"""
4090:
172.20.163.193:800
192.168.1.2: 22
yuanyn: yuanyongna

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
"""

"""
conda create -n shw_DTPharMol python=3.10
conda activate shw_DTPharMol

# https://pytorch.org/get-started/previous-versions/
# CUDA 10.2
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
# CUDA 12.2
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

conda install conda-forge::bert_score
pip install bert_score

conda install conda-forge::blobfile
pip install blobfile

conda install conda-forge::nltk
pip install nltk

conda install conda-forge::numpy
pip install numpy==1.26.4 (2.2.6 --> 1.26.4)
pip install "numpy<2.0" --force-reinstall
pip install "numpy<2.0" --no-cache-dir

conda install conda-forge::packaging
pip install packaging (Requirement already satisfied)

pip install psutil

conda install conda-forge::pyyaml
pip install PyYAML (Requirement already satisfied)

conda install conda-forge::setuptools
pip install setuptools (Requirement already satisfied)

conda install conda-forge::spacy
pip install spacy

conda install conda-forge::torchmetrics
pip install torchmetrics

conda install conda-forge::tqdm
pip install tqdm (Requirement already satisfied)

conda install conda-forge::transformers
pip install transformers (Requirement already satisfied)

conda install conda-forge::wandb
pip install wandb

conda install conda-forge::datasets
pip install datasets (fsspec-2024.6.1)

conda install -c rdkit rdkit
pip install rdkit

pip install nvitop

conda install dglteam::dgl-cuda10.2
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html

conda install openbabel -c conda-forge

conda install conda-forge::vina
conda install -c conda-forge autodock-vina
conda install -c conda-forge vina
vina --version

conda install pyg -c pyg

conda install seaborn

pip install setproctitle
"""


import os
import sys
import psutil
import torch
import time
import setproctitle
import importlib.metadata

# 常量定义
SEPARATOR = "\n" + "#" * 100 + "\n"
MATRIX_SIZE = 3000
NUM_ITERATIONS = 100


def get_process_info():
    """获取并格式化进程信息"""
    setproctitle.setproctitle("envs")
    current_process = psutil.Process(os.getpid())
    print(
        f"Process Info:\n"
        f"PID: {current_process.pid}\n"
        f"Name: {current_process.name()}\n"
        f"Create Time: {time.ctime(current_process.create_time())}\n"
        f"Statue: {current_process.status()}\n"
        f"Memory: {current_process.memory_info()}"
    )


def check_gpu_availability():
    """获取 GPU 信息"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")
    print(
        f"PyTorch Environment Info:\n"
        f"Python Version: {sys.version}\n"
        f"CUDA Version: {torch.version.cuda}\n"
        f"CUDA Availability: {torch.cuda.is_available()}\n"
        f"CUDNN Version: {torch.backends.cudnn.version()}\n"
        f"CUDNN Availability: {torch.backends.cudnn.is_available()}\n"
        f"PyTorch Version: {torch.__version__}\n"
        f"GPU Availability: {torch.cuda.is_available()}\n"
        f"GPU Count: {torch.cuda.device_count()}\n"
        f"Current GPU Model: {torch.cuda.get_device_name(0)}\n"
        f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}\n"
        f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB\n"
        f"VRAM Utilization: {torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100:.1f}%\n"
        f"TensorCore Support: {torch.cuda.get_device_properties(0).major >= 7}\n"
        f"BF16 Support: {torch.cuda.is_bf16_supported()}"
    )


def benchmark_gpu_performance():
    """执行 GPU/CPU 性能基准测试"""
    # 初始化测试矩阵 (使用非对称尺寸避免缓存优化影响)
    a = torch.randn(MATRIX_SIZE + 128, MATRIX_SIZE)
    b = torch.randn(MATRIX_SIZE, MATRIX_SIZE + 64)
    # 预热阶段优化 (执行多次空操作)
    for _ in range(3):  # 消除驱动初始化延迟
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    # cpu 计算时间
    start_time = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        _ = torch.matmul(a, b)
    cpu_time = (time.perf_counter() - start_time) * 1000
    # gpu 计算时间
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    a, b = a.to("cuda"), b.to("cuda")
    with torch.no_grad():
        for _ in range(NUM_ITERATIONS):
            _ = torch.matmul(a, b)
    end_event.record()
    torch.cuda.synchronize()
    gpu_time = start_event.elapsed_time(end_event)
    print(
        "Performance Benchmark:\n"
        f"CPU Time: {cpu_time:.2f} ms\n"
        f"GPU Time: {gpu_time:.2f} ms\n"
        f"Speedup Ratio: {cpu_time / gpu_time:.1f}x"
    )


def get_installed_packages():
    """获取已安装的软件包列表"""
    installed_packages = importlib.metadata.distributions()
    for package in installed_packages:
        print(f"{package.metadata['Name']} {package.metadata['Version']}")


def envs_create():
    # 获取进程信息
    print(SEPARATOR)
    get_process_info()

    # 获取 GPU 信息
    print(SEPARATOR)
    check_gpu_availability()

    # 性能基准测试
    print(SEPARATOR)
    benchmark_gpu_performance()

    # 软件包列表
    print(SEPARATOR)
    get_installed_packages()

    print(SEPARATOR)
    print("Environment Validation Passed!")


if __name__ == "__main__":
    envs_create()
