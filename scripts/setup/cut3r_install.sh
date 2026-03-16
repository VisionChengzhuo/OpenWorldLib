#!/bin/bash
# scripts/setup/cut3r_install.sh
# Description: Setup environment for cut3r-based experiments in SceneFlow
# Usage: bash scripts/setup/cut3r_install.sh

echo "=== [1/3] Installing the base environment ==="
pip install torch==2.5.1 torchvision torchaudio

echo "=== [2/3] Installing the requirements (3d_ply_default extra) ==="
pip install -e ".[3d_ply_default]"

echo "=== [3/3] Installing additional cut3r-specific dependencies ==="
pip install "numpy==1.26.4" roma opencv-python scipy trimesh tensorboard "pyglet<2" "huggingface-hub[torch]>=0.22" viser lpips hydra-core pillow==10.3.0 h5py accelerate gradio einops matplotlib tqdm "scikit-learn" simple_knn

echo "=== Setup completed! ==="
