# TurtleBot3 Left Wheel Troubleshooting Guide

## Diagnostic Test Results

You reported: **Left wheel is not working**

The diagnostic test just ran 6 tests. Please observe:

### What to Check During Each Test:

**TEST 1: Forward** - Both wheels should move forward
- ❓ Did the left wheel move at all?

**TEST 2: Rotate Left** - Right wheel forward, Left wheel backward
- ❓ Did the left wheel move backward?

**TEST 3: Rotate Right** - Left wheel forward, Right wheel backward
- ❓ Did the left wheel move forward?

**TEST 4: Backward** - Both wheels should move backward
- ❓ Did the left wheel move backward?

**TEST 5 & 6: Arcs** - Mixed movements
- ❓ Any left wheel movement?

---

## Common Causes & Solutions

### 1. **Motor Connection Issue** (Most Common)

**Check on TurtleBot (192.168.0.18):**

SSH into TurtleBot:
```bash
ssh ubuntu@192.168.0.18
```

Check motor connections to OpenCR board:
```bash
# Look at joint states to see if motors are detected
ros2 topic echo /joint_states --once
```

You should see both `wheel_left_joint` and `wheel_right_joint`. If left is missing or always 0, it's a connection issue.

**Physical Check:**
1. Power off the TurtleBot completely
2. Open the robot (remove top plate if needed)
3. Check the cable from left motor to OpenCR board
4. Ensure it's firmly connected
5. Check for damaged wires

---

### 2. **Motor Driver Failure**

If the motor is connected but not working:
- The motor driver chip on OpenCR board may be damaged
- This requires OpenCR board replacement or repair

---

### 3. **Physical Obstruction**

Check if wheel can spin freely:
1. Power off the robot
2. Try to manually rotate the left wheel
3. It should spin freely
4. Check for:
   - Hair/thread wrapped around axle
   - Debris blocking wheel
   - Damaged wheel mount

---

### 4. **Check Battery Level**

Low battery can cause one motor to fail:
```bash
ros2 topic echo /battery_state --once
```

Look at the voltage. Should be above 11V for normal operation.

---

### 5. **Check OpenCR Firmware**

The OpenCR board might need firmware update:
```bash
# On TurtleBot
cd ~/OpenCR
./update.sh
```

---

### 6. **Test Motor Directly**

On the TurtleBot, you can test if the OpenCR board sees the motor:

```bash
# Check if dynamixel SDK detects both motors
ros2 topic echo /sensor_state
```

Look for `left_encoder` values. If they're always 0 or not changing, the motor isn't responding.

---

## Quick Test Commands

From your host computer (192.168.0.10):

**Test forward (both wheels):**
```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}, angular: {z: 0.0}}"
```

**Test pure rotation left (should show if left wheel can move backward):**
```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.5}}"
```

**Test pure rotation right (should show if left wheel can move forward):**
```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: -0.5}}"
```

---

## SSH Into TurtleBot for Hardware Checks

```bash
ssh ubuntu@192.168.0.18
```

Once logged in, check:

1. **Motor status:**
   ```bash
   ros2 topic echo /joint_states
   ```

2. **Sensor state:**
   ```bash
   ros2 topic echo /sensor_state
   ```

3. **System logs:**
   ```bash
   journalctl -u turtlebot3-robot.service -n 50
   ```

---

## Next Steps

**Please tell me:**
1. ✅ During the 6 tests, did the left wheel move AT ALL in ANY test?
2. 🔊 Do you hear the left motor making any sound?
3. 🔋 What's the battery voltage? (check /battery_state)
4. 🔌 Can you check if the left motor cable is properly connected?

This will help me narrow down the exact problem!
