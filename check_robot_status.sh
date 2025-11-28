#!/bin/bash
# Check robot status and logs

echo "Checking TurtleBot status..."
echo ""

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
export ROS_LOCALHOST_ONLY=0

echo "1. Node list:"
ros2 node list

echo ""
echo "2. /cmd_vel topic info:"
ros2 topic info /cmd_vel

echo ""
echo "3. Listening to cmd_vel for 3 seconds:"
timeout 3 ros2 topic echo /cmd_vel || echo "No data received"

echo ""
echo "4. Checking if turtlebot3_node is responding:"
ros2 node info /turtlebot3_node | head -20
