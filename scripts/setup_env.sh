#!/bin/bash
# CalSem 服务器端环境安装（目标: /home/wchan/.virtualenvs/CalSem, Python 3.10, GPU cc=8.9）
set -e
VPY=/home/wchan/.virtualenvs/CalSem/bin/python

echo "=== [1/4] upgrade pip ==="
$VPY -m pip install --upgrade pip

echo "=== [2/4] torch cu124 ==="
$VPY -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo "=== [3/4] diffusers/transformers ==="
$VPY -m pip install diffusers==0.40.0 accelerate==1.15.0 safetensors==0.8.0
$VPY -m pip install transformers==5.17.0 || $VPY -m pip install transformers

echo "=== [4/4] misc ==="
$VPY -m pip install opencv-python numpy pillow pyyaml scikit-learn tqdm

echo "=== verify ==="
$VPY -c "import torch, torchvision, diffusers, transformers, accelerate, safetensors, cv2, sklearn, yaml; print('TORCH', torch.__version__, 'CUDA', torch.cuda.is_available()); print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
echo "SETUP_DONE"
