# TurtleBot3 Manual Control Testing

This guide helps you test the TurtleBot3 movement without the camera/vision system.

## Prerequisites

Make sure the TurtleBot3 robot base is powered on and connected.

## Setup

### 1. Set ROS Domain ID (if needed)
```bash
export ROS_DOMAIN_ID=17
```

### 2. Check if TurtleBot is Running

On the TurtleBot (Raspberry Pi), the robot base should be running. If not, start it:

```bash
# SSH into TurtleBot
ssh ubuntu@<TURTLEBOT_IP>

# Source ROS 2
source /opt/ros/humble/setup.bash

# Bring up the robot
ros2 launch turtlebot3_bringup robot.launch.py
```

### 3. Verify Connection from Host Computer

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17

# Check if /cmd_vel topic is available
ros2 topic list | grep cmd_vel

# Monitor velocity commands
ros2 topic echo /cmd_vel
```

## Testing Options

### Option 1: Automatic Movement Test (Recommended for First Test)

This runs a pre-programmed sequence of movements to verify the robot responds:

```bash
cd /home/turtlebot/vision360
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
python3 test_movement.py
```

**Test Sequence:**
1. Move forward for 2 seconds
2. Move backward for 2 seconds
3. Rotate left for 2 seconds
4. Rotate right for 2 seconds
5. Move in a circle for 3 seconds
6. Stop

### Option 2: Manual Keyboard Control (Interactive)

Control the robot manually with keyboard:

```bash
cd /home/turtlebot/vision360
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
python3 manual_control.py
```

**Controls:**
- `W` or `↑` : Move Forward
- `S` or `↓` : Move Backward
- `A` or `←` : Turn Left
- `D` or `→` : Turn Right
- `X` : Stop
- `+` : Increase Speed
- `-` : Decrease Speed
- `Q` : Quit

### Option 3: Direct ROS 2 Command Test (Quick Check)

Send a single command to test:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17

# Move forward slowly
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}, angular: {z: 0.0}}"

# Stop
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}"
```

## Troubleshooting

### Robot doesn't move

1. **Check if robot base is running:**
   ```bash
   ros2 topic list | grep cmd_vel
   ```
   You should see `/cmd_vel` in the list.

2. **Check ROS_DOMAIN_ID matches on both devices:**
   ```bash
   echo $ROS_DOMAIN_ID
   ```

3. **Verify network connectivity:**
   ```bash
   ping <TURTLEBOT_IP>
   ```

4. **Check if commands are being received:**
   ```bash
   ros2 topic echo /cmd_vel
   ```
   Run this while sending commands from another terminal.

### Robot moves but behaves strangely

- Check battery level on TurtleBot
- Ensure robot is on a flat surface with enough space
- Reduce speed using the `-` key in manual control

### Permission errors with keyboard input

The manual control script needs raw terminal input. Run it directly in the terminal (not through an IDE).

## Safety Notes

- **Always have clear space** around the robot (at least 2m radius)
- **Be ready to press 'X' or 'Q'** to stop the robot
- Start with slow speeds and gradually increase
- Keep the emergency stop button accessible if your TurtleBot has one

## Next Steps

Once movement is verified:
1. Test the camera separately
2. Run the full vision360 system in ROS mode
3. Calibrate the vision system parameters in `config/config.yaml`
