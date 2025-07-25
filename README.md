# Machine_AI_Setup

## 1.
#!/bin/bash

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo "=== Step 1: Installing CUDA 12.8 ==="
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

echo "=== Step 2: Install cuDNN 8.9.7.29 for CUDA 12.8 ==="
# You must have downloaded cuDNN .deb files from https://developer.nvidia.com/cudnn and placed them in this directory
# Adjust the file name if you have a newer version
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/cudnn-local-08A7D361-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y libcudnn8 libcudnn8-dev libcudnn8-samples

echo "=== Step 3: Set Environment Variables for CUDA 12.8 ==="
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo "=== Step 4: Create Conda Env and Install PyTorch Nightly with CUDA 12.8 ==="
conda create -n rtx5080_env python=3.10 -y
conda activate rtx5080_env

pip install --upgrade pip
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo "=== Step 5: Install Ultralytics and xformers ==="
pip install ultralytics
pip install --pre xformers -f https://download.pytorch.org/whl/nightly/cu128

echo "=== Step 6: Test CUDA Access in PyTorch ==="
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
