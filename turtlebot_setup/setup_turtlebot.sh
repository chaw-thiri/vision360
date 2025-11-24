#!/bin/bash
# TurtleBot3 Setup Script for Raspberry Pi
# Run this on the TurtleBot's Raspberry Pi

set -e

echo "=========================================="
echo "TurtleBot3 Camera Setup for ROS 2 Humble"
echo "=========================================="

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo "Warning: This doesn't appear to be a Raspberry Pi"
fi

# Update system
echo "[1/6] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Pi Camera dependencies
echo "[2/6] Installing Pi Camera dependencies..."
sudo apt install -y \
    python3-picamera2 \
    python3-libcamera \
    libcamera-apps \
    python3-opencv \
    python3-numpy

# Enable camera interface
echo "[3/6] Enabling camera interface..."
sudo raspi-config nonint do_camera 0 2>/dev/null || echo "Camera already enabled or using newer Pi OS"

# Install ROS 2 camera packages
echo "[4/6] Installing ROS 2 camera packages..."
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-compressed-image-transport

# Set ROS_DOMAIN_ID
echo "[5/6] Configuring ROS 2 environment..."
ROS_DOMAIN_ID=17

# Add to bashrc if not already there
if ! grep -q "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# ROS 2 Domain ID for TurtleBot" >> ~/.bashrc
    echo "export ROS_DOMAIN_ID=$ROS_DOMAIN_ID" >> ~/.bashrc
fi

# Source ROS 2
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
fi

# Create systemd service for camera streamer (optional)
echo "[6/6] Creating systemd service (optional)..."

sudo tee /etc/systemd/system/camera-streamer.service > /dev/null << 'EOF'
[Unit]
Description=TurtleBot Camera Streamer
After=network.target

[Service]
Type=simple
User=ubuntu
Environment="ROS_DOMAIN_ID=17"
ExecStart=/bin/bash -c "source /opt/ros/humble/setup.bash && python3 /home/ubuntu/capstone/turtlebot_setup/camera_streamer.py"
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start camera streamer manually:"
echo "  source /opt/ros/humble/setup.bash"
echo "  export ROS_DOMAIN_ID=17"
echo "  python3 camera_streamer.py"
echo ""
echo "To enable auto-start on boot:"
echo "  sudo systemctl enable camera-streamer"
echo "  sudo systemctl start camera-streamer"
echo ""
echo "Please reboot for all changes to take effect."
