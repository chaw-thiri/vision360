#!/bin/bash
# Quick launcher for manual control

echo "=================================================="
echo "TurtleBot3 Manual Control"
echo "=================================================="
echo ""

# Source ROS 2
source /opt/ros/humble/setup.bash

# Set domain ID (change if needed)
export ROS_DOMAIN_ID=17

echo "Checking ROS 2 connection..."
if ros2 topic list | grep -q "cmd_vel"; then
    echo "✓ Robot detected! /cmd_vel topic found"
else
    echo "⚠ Warning: /cmd_vel topic not found"
    echo "  Make sure TurtleBot robot base is running"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Starting manual control..."
echo "Use WASD or arrow keys to move, Q to quit"
echo ""

cd /home/turtlebot/vision360
python3 manual_control.py
