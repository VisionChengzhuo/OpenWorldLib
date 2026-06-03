#!/bin/bash
# Sana-WM environment setup for OpenWorldLib
# This script configures the conda environment for Sana-WM inference.
# Usage: bash scripts/setup/sana_wm_install.sh

set -e

echo "=== Sana-WM OpenWorldLib Setup ==="
echo "Make sure you have the base conda environment for OpenWorldLib activated."
echo ""
echo "Required pip packages (in addition to OpenWorldLib base):"
echo "  pip install pyrallis imageio safetensors"
echo "  pip install \"diffusers>=0.32.0\"  # for LTX-2 support"
echo "  pip install \"transformers>=4.47.0\"  # for Gemma-2-2b-it"
echo "  pip install scipy  # for trajectory interpolation"
echo ""

# xformers compatibility check
echo "IMPORTANT: DISABLE_XFORMERS=1 will be set automatically on import."
echo "Sana-WM requires torch >= 2.x with native SDPA support."
echo ""
echo "Environment setup complete."