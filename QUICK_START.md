# Vision360 Quick Start Guide

## One-Command Startup!

### Step 1: On TurtleBot (192.168.0.18)

```bash
ssh turtlebot@192.168.0.18
cd ~/vision360
./start_turtlebot.sh
```

**This automatically:**
- ✅ Starts robot base
- ✅ Enables motor power
- ✅ Starts camera streaming
- ✅ Shows status and PIDs

**Keep this terminal open!**

---

### Step 2: On Your Desktop/VM (192.168.0.10)

```bash
cd /home/turtlebot/vision360
./start_vision_desktop.sh
```

**This automatically:**
- ✅ Checks TurtleBot connection
- ✅ Checks camera feed
- ✅ Safety confirmation
- ✅ Starts vision system

**That's it!** 🎉

---

## Manual Startup (If Scripts Don't Work)

### TurtleBot Terminal 1:
```bash
ssh turtlebot@192.168.0.18
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_bringup robot.launch.py
```

### TurtleBot Terminal 2:
```bash
ssh turtlebot@192.168.0.18
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
ros2 service call /motor_power std_srvs/srv/SetBool "{data: true}"

cd ~/vision360
python3 fix_camera_streamer.py
```

### Desktop Terminal:
```bash
cd /home/turtlebot/vision360
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
export ROS_LOCALHOST_ONLY=0
python3 main.py --mode ros
```

---

## Stopping Everything

### Quick Stop:
Press **Ctrl+C** in the TurtleBot terminal, then press **Q** in the vision window.

### Emergency Stop:
Press **S** key in the vision window anytime!

---

## Troubleshooting

### Robot not moving?
```bash
# On TurtleBot, enable motors:
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
ros2 service call /motor_power std_srvs/srv/SetBool "{data: true}"
```

### Can't see camera?
```bash
# On TurtleBot, restart camera:
pkill -f camera_streamer
cd ~/vision360
python3 fix_camera_streamer.py
```

### Connection issues?
```bash
# Check network:
ping 192.168.0.18

# Check ROS topics:
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
export ROS_LOCALHOST_ONLY=0
ros2 topic list
```

Should see:
- `/cmd_vel`
- `/camera/image_raw/compressed`
- `/odom`
- `/scan`

---

## What the Robot Does

🚶 **Stops/slows** when detecting people
🛣️ **Follows lanes** when detected
🚦 **Stops at red**, slows at yellow, goes on green
🎯 **Makes real-time decisions** based on vision

---

## System Architecture

```
TurtleBot (192.168.0.18)          Desktop (192.168.0.10)
├─ robot.launch.py                ├─ main.py --mode ros
├─ fix_camera_streamer.py         │  ├─ Receives camera feed
└─ Motors + Camera                │  ├─ Runs YOLOv8 detection
                                  │  ├─ Lane detection
                                  │  ├─ Traffic lights
                                  │  ├─ Decision making
                                  └─ Sends /cmd_vel commands
```

All communication over ROS 2 (DOMAIN_ID=17)
