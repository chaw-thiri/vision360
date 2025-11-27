#!/bin/bash
# Automated Vision System Startup - Run on Desktop/VM

echo "=========================================="
echo "Vision360 Desktop Startup"
echo "=========================================="
echo ""

# Setup environment
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
export ROS_LOCALHOST_ONLY=0

# Check if TurtleBot is ready
echo "Checking TurtleBot connection..."

if ! ros2 topic list | grep -q "/cmd_vel"; then
    echo "✗ TurtleBot not detected!"
    echo ""
    echo "Make sure the TurtleBot startup script is running:"
    echo "  ssh turtlebot@192.168.0.18"
    echo "  cd ~/vision360"
    echo "  ./start_turtlebot.sh"
    echo ""
    exit 1
fi

echo "✓ Robot base connected"

if ! ros2 topic list | grep -q "/camera/image_raw/compressed"; then
    echo "⚠ Camera not detected!"
    echo "  Make sure camera streamer is running on TurtleBot"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Camera connected"
fi

echo ""
echo "=========================================="
echo "⚠ SAFETY CHECK"
echo "=========================================="
echo "Before starting:"
echo "  • Robot has 2m clear space"
echo "  • You can reach keyboard for emergency stop"
echo "  • Battery is charged"
echo ""
read -p "Ready to start? (yes/no): " response

if [ "$response" != "yes" ]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "Starting Vision360 System..."
echo "Controls:"
echo "  S = Emergency Stop"
echo "  R = Resume"
echo "  Q = Quit"
echo "  D = Debug view"
echo ""
sleep 2

cd /home/turtlebot/vision360
python3 main.py --mode ros
