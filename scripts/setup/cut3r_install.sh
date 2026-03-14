#!/bin/bash
# scripts/setup/cut3r_install.sh
# Description: Setup environment for cut3r installation of SceneFlow
# Usage: bash scripts/setup/cut3r_install.sh

echo "=== [1/2] Installing the base environment ==="
pip install "huggingface-hub[torch]>=0.22"
pip install pillow==10.3.0

echo "=== [2/2] Installing additional dependencies ==="
pip install scipy trimesh "pyglet<2" viser lpips h5py scikit-learn simple_knn

echo "=== Setup completed! ==="
