#!/bin/bash
# 为 WSL 安装支持 GPU 的 PyTorch 和依赖

echo "=========================================="
echo "Installing GPU-enabled dependencies"
echo "=========================================="

cd /mnt/d/Project/WY20251027/code

# 激活虚拟环境
source venv_wsl/bin/activate

# 卸载 CPU 版本的 PyTorch
echo "Uninstalling CPU version of PyTorch..."
pip uninstall torch torchvision torchaudio -y

# 安装 CUDA 版本的 PyTorch
echo "Installing CUDA version of PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=========================================="
echo "Installation complete!"
echo "=========================================="

# 验证 CUDA 是否可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
