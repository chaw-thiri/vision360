#!/bin/bash
# Quick robot debug script - Run on TurtleBot

echo "=========================================="
echo "TurtleBot Hardware Debug"
echo "=========================================="
echo ""

echo "1. Checking OpenCR USB connection:"
ls -l /dev/ttyACM* 2>/dev/null || echo "  ❌ No OpenCR found (/dev/ttyACM*)"
echo ""

echo "2. Checking user permissions:"
groups | grep -q dialout && echo "  ✓ User in dialout group" || echo "  ❌ User NOT in dialout group"
echo ""

echo "3. Checking if TurtleBot node is running:"
ros2 node list | grep turtlebot3_node && echo "  ✓ Node running" || echo "  ❌ Node not found"
echo ""

echo "4. Checking ROS topics:"
echo "  /cmd_vel: $(ros2 topic info /cmd_vel 2>/dev/null | grep 'Subscription count' || echo 'NOT FOUND')"
echo "  /joint_states: $(ros2 topic list | grep joint_states || echo 'NOT PUBLISHED')"
echo "  /odom: $(ros2 topic list | grep odom || echo 'NOT PUBLISHED')"
echo ""

echo "5. Testing OpenCR connection:"
if [ -e /dev/ttyACM0 ]; then
    echo "  Trying to read from OpenCR..."
    timeout 2 cat /dev/ttyACM0 >/dev/null 2>&1 && echo "  ✓ OpenCR responding" || echo "  ⚠ OpenCR not responding"
fi
echo ""

echo "=========================================="
echo "Common Issues:"
echo "=========================================="
echo "1. OpenCR not connected: Reconnect USB cable"
echo "2. Permission issue: sudo usermod -aG dialout \$USER (then logout/login)"
echo "3. OpenCR needs reset: Press reset button on OpenCR"
echo "4. Wrong firmware: Flash OpenCR with TurtleBot firmware"
echo ""
echo "To restart everything:"
echo "  1. Stop robot.launch.py (Ctrl+C)"
echo "  2. Unplug/replug OpenCR USB"
echo "  3. Wait 5 seconds"
echo "  4. Run: ros2 launch turtlebot3_bringup robot.launch.py"
