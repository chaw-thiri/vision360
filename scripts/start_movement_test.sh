#!/bin/bash
# Quick launcher for automated movement test

echo "=================================================="
echo "TurtleBot3 Automated Movement Test"
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
echo "⚠ SAFETY WARNING:"
echo "  - Ensure robot has 2m clearance in all directions"
echo "  - Be ready to press Ctrl+C to stop"
echo ""
read -p "Ready to start test? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

echo ""
echo "Starting movement test in 3 seconds..."
sleep 1
echo "3..."
sleep 1
echo "2..."
sleep 1
echo "1..."
sleep 1

cd /home/turtlebot/vision360
python3 tests/test_movement.py
