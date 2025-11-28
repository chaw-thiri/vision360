#!/bin/bash
# Automated TurtleBot Startup Script
# Run this on the TurtleBot (192.168.0.18)

echo "=========================================="
echo "TurtleBot Vision360 Startup"
echo "=========================================="
echo ""

# Setup environment
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
export TURTLEBOT3_MODEL=burger

# Kill any existing processes
echo "Cleaning up old processes..."
pkill -f robot.launch.py 2>/dev/null
pkill -f camera_streamer 2>/dev/null
sleep 2

# Start robot base in background
echo "Starting robot base..."
ros2 launch turtlebot3_bringup robot.launch.py > /tmp/robot_launch.log 2>&1 &
ROBOT_PID=$!

# Wait for robot to initialize
echo "Waiting for robot initialization..."
sleep 8

# Enable motor power
echo "Enabling motor power..."
ros2 service call /motor_power std_srvs/srv/SetBool "{data: true}" >/dev/null 2>&1
sleep 1

# Check if robot is running
if ps -p $ROBOT_PID > /dev/null; then
    echo "✓ Robot base started (PID: $ROBOT_PID)"
else
    echo "✗ Robot base failed to start!"
    echo "Check logs: cat /tmp/robot_launch.log"
    exit 1
fi

# Start camera streamer
echo "Starting camera streamer..."
cd ~/vision360
python3 fix_camera_streamer.py > /tmp/camera_streamer.log 2>&1 &
CAMERA_PID=$!
sleep 3

if ps -p $CAMERA_PID > /dev/null; then
    echo "✓ Camera streamer started (PID: $CAMERA_PID)"
else
    echo "✗ Camera streamer failed to start!"
    echo "Check logs: cat /tmp/camera_streamer.log"
fi

echo ""
echo "=========================================="
echo "TurtleBot Ready!"
echo "=========================================="
echo "Robot base PID: $ROBOT_PID"
echo "Camera streamer PID: $CAMERA_PID"
echo ""
echo "To stop:"
echo "  kill $ROBOT_PID $CAMERA_PID"
echo ""
echo "To view logs:"
echo "  tail -f /tmp/robot_launch.log"
echo "  tail -f /tmp/camera_streamer.log"
echo ""
echo "Now run the vision system on your desktop!"
echo "=========================================="

# Keep script running so processes stay alive
echo ""
echo "Press Ctrl+C to stop everything..."
trap "echo 'Stopping...'; kill $ROBOT_PID $CAMERA_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
