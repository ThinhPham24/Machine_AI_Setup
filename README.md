# Machine AI Setup: RTX 5080 (Blackwell)

![Platform](https://img.shields.io/badge/OS-Ubuntu_22.04-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Nightly_2.7.0-red)
![GPU](https://img.shields.io/badge/GPU-RTX_5080-76b900)

This repository documents the environment setup and installation steps for Deep Learning on the **NVIDIA RTX 5080 (Blackwell Architecture)**.

> **⚠️ Note:** The RTX 5080 requires the latest **CUDA 12.8** and **PyTorch Nightly** builds. Stable releases may not yet support the Blackwell architecture.

## 🛠 Prerequisites

| Component | Version | Description |
| :--- | :--- | :--- |
| **GPU** | RTX 5080 | [Blackwell Architecture Info](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) |
| **OS** | Ubuntu 22.04 | LTS (x86_64) |
| **CUDA Toolkit** | 12.8.0 | [Download Link](https://developer.nvidia.com/cuda-12-8-0-download-archive) |
| **cuDNN** | 9.11.0 | [Download Link](https://developer.nvidia.com/cudnn-downloads) |

---

## 🚀 Installation Guide

### 1. System-Level Setup (CUDA & cuDNN)

First, remove old CUDA versions and install the new 12.8 toolkit and cuDNN 9.11 drivers.

```bash
# 1. Install CUDA 12.8
wget [https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-1_amd64.deb](https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-1_amd64.deb)
sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# 2. Install cuDNN 9.11.0
# Ensure you have downloaded the .deb file from the NVIDIA Developer portal
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.11.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.11.0/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y libcudnn9 libcudnn9-dev libcudnn9-samples

# 3. Configure Environment Variables
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. Python Environment (Conda)
We use Conda to manage dependencies and install the PyTorch Preview (Nightly) build which contains the necessary PTX/SASS support for Blackwell GPUs.

```bash
# Create and activate environment
conda create -n rtx5080_env python=3.10 -y
conda activate rtx5080_env

# Upgrade pip
pip install --upgrade pip

# Install PyTorch Nightly (CUDA 12.8)
pip install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu128](https://download.pytorch.org/whl/nightly/cu128)
```

### 3. Verification
Run the following command to verify that PyTorch can see the RTX 5080 and that CUDA is properly linked.
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```


