#!/bin/bash
# ROS 2 Humble Setup Script for macOS
# This script helps set up ROS 2 on MacBook for the vision system

set -e

echo "=========================================="
echo "ROS 2 Humble Setup for macOS"
echo "=========================================="

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "[1/5] Installing dependencies via Homebrew..."
brew install python@3.11 cmake pkg-config wget

# Create virtual environment
echo "[2/5] Creating Python virtual environment..."
VENV_PATH="$HOME/.venvs/ros2_vision"
python3.11 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# Install Python dependencies
echo "[3/5] Installing Python packages..."
pip install --upgrade pip
pip install \
    opencv-python>=4.8.0 \
    numpy>=1.24.0 \
    ultralytics>=8.0.0 \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    PyYAML>=6.0 \
    scipy>=1.11.0 \
    Pillow>=10.0.0

# ROS 2 on macOS options
echo "[4/5] ROS 2 Installation Options..."
echo ""
echo "ROS 2 on macOS requires either:"
echo "  A) Docker with ROS 2 image (recommended)"
echo "  B) Build from source (complex)"
echo "  C) Use without ROS (standalone mode)"
echo ""

# Download YOLOv8 model
echo "[5/5] Downloading YOLO model..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment created at: $VENV_PATH"
echo ""
echo "To activate the environment:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "For ROS 2 communication with TurtleBot:"
echo ""
echo "Option A - Docker (Recommended):"
echo "  docker run -it --rm --net=host -e ROS_DOMAIN_ID=17 ros:humble"
echo ""
echo "Option B - Standalone mode (no ROS):"
echo "  python main.py --mode standalone"
echo ""
echo "Set ROS_DOMAIN_ID to match TurtleBot:"
echo "  export ROS_DOMAIN_ID=17"
