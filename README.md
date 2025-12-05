# Vision360 - Autonomous TurtleBot3 Vision System

Real-time computer vision system for TurtleBot3 featuring pedestrian detection, lane following, and traffic sign recognition.

---

## 🚀 Quick Start - Running the Vision System

The vision system requires **TWO files** running simultaneously:

### 📦 Main Components

```
vision360/
├── main.py                      ⭐ Vision processing (runs on desktop)
└── scripts/
    └── start_turtlebot.sh       ⭐ Robot control (runs on TurtleBot)
```

### 🤖 Step 1: Start TurtleBot (On Robot via SSH)

```bash
# SSH to TurtleBot
ssh turtlebot@192.168.0.18

# Navigate to project
cd ~/vision360

# Start robot base and camera
./scripts/start_turtlebot.sh
```

**This starts:**
- ✓ Robot motors (OpenCR controller)
- ✓ Camera streaming to ROS topics

---

### 💻 Step 2: Start Vision System (On Your Desktop/Laptop)

**Option A: Direct command**
```bash
python3 main.py --mode ros
```

**Option B: Automated script (recommended)**
```bash
./scripts/start_vision_desktop.sh
```

**This runs:**
- ✓ Person detection (YOLOv8)
- ✓ Lane detection
- ✓ Traffic sign/light detection
- ✓ Autonomous navigation decisions
- ✓ Sends velocity commands to robot

---

## 📋 Complete Setup Command Summary

```bash
# ==========================================
# Terminal 1 - TurtleBot (Raspberry Pi)
# ==========================================
ssh turtlebot@192.168.0.18
cd ~/vision360
./scripts/start_turtlebot.sh

# ==========================================
# Terminal 2 - Desktop (Your Computer)
# ==========================================
cd ~/vision360
python3 main.py --mode ros
```

**That's it!** The robot will now drive autonomously following lanes, detecting pedestrians, and responding to traffic signs.

---

## ⌨️ Controls While Running

| Key | Action |
|-----|--------|
| `Q` | Emergency stop |
| `R` | Resume autonomous driving |
| `X` | Quit application |

---

## 🎥 Testing Modes (Without TurtleBot)

You can test the vision system without a TurtleBot using the automated script:

### Test with Webcam
```bash
# Using script (recommended)
./scripts/start_vision_desktop.sh webcam

# Or direct command
python3 main.py --mode webcam --camera 0
```

### Test with Video File
```bash
# Using script (recommended)
./scripts/start_vision_desktop.sh video data/road_test.mp4

# Or direct command
python3 main.py --mode video --input data/road_test.mp4
```

### Script Usage Summary
```bash
./scripts/start_vision_desktop.sh                    # ROS mode (default, requires TurtleBot)
./scripts/start_vision_desktop.sh ros                # ROS mode (explicit)
./scripts/start_vision_desktop.sh webcam             # Webcam mode (no robot needed)
./scripts/start_vision_desktop.sh video <file.mp4>   # Video mode (no robot needed)
```

---

## 📦 Installation (First Time Setup)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install ROS 2 Humble (Ubuntu 22.04)

```bash
# Add ROS 2 repository
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2
sudo apt update
sudo apt install -y ros-humble-desktop \
                    ros-humble-cv-bridge \
                    ros-humble-image-transport

# Configure environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=17" >> ~/.bashrc
source ~/.bashrc
```

### 3. Verify Setup

```bash
# Check ROS 2 installation
ros2 topic list

# Should see topics if TurtleBot is running:
# /camera/image_raw/compressed
# /cmd_vel
```

---

## 🎯 What Each Mode Does

| Mode | Model Used | Use Case |
|------|------------|----------|
| `--mode ros` | traffic6.pt | Real TurtleBot operation |
| `--mode webcam` | traffic6.pt | Testing with USB camera |
| `--mode video` | traffic_sign_lights.pt | Testing with video files |

*The system automatically selects the appropriate traffic sign detection model based on the mode.*

---

## 🔧 Configuration

Edit `config/config.yaml` to adjust:

```yaml
# Robot speeds
turtlebot:
  normal_speed: 0.06      # m/s (default driving speed)
  max_linear_velocity: 0.08
  max_angular_velocity: 1.5

# Detection sensitivity
person_detection:
  confidence_threshold: 0.5
  danger_zone_ratio: 0.4

# Lane detection colors (adjust for your track)
lane_detection:
  yellow_tape: { h_low: 15, s_low: 80, v_low: 80 }
  black_line: { v_high: 70 }
```

---

## 🏗️ Project Structure

```
vision360/
├── main.py                    ⭐ Main vision processing
├── scripts/
│   ├── start_turtlebot.sh     ⭐ TurtleBot launcher
│   └── start_vision_desktop.sh ⭐ Desktop launcher
├── config/
│   └── config.yaml            ⚙️ Configuration settings
├── models/
│   ├── yolov8n.pt             🤖 Person detection model
│   ├── traffic6.pt            🚦 Traffic signs (webcam/ROS)
│   └── traffic_sign_lights.pt 🚦 Traffic signs (video)
├── src/
│   ├── detectors/             🔍 Detection modules
│   ├── controller/            🎮 Navigation logic
│   └── ros_nodes/             📡 ROS integration
├── tests/                     🧪 Test scripts
├── docs/                      📖 Documentation
└── requirements.txt           📦 Dependencies
```

---

## 🎓 System Features

- ✅ **Person Detection** - YOLOv8-based pedestrian detection with emergency stop
- ✅ **Lane Following** - HSV color detection + Hough transform for lane tracking
- ✅ **Traffic Signs** - YOLO-based traffic light and road sign recognition
- ✅ **Boundary Detection** - Black platform detection for obstacle avoidance
- ✅ **PID Control** - Smooth lane following with velocity smoothing
- ✅ **Manual Override** - Keyboard control for testing
- ✅ **ROS 2 Integration** - Full TurtleBot3 Humble support

---

## 📚 Additional Documentation

- **[DETAILED_README.md](DETAILED_README.md)** - Complete technical documentation, algorithms, and troubleshooting
- **[docs/QUICK_START.md](docs/QUICK_START.md)** - Step-by-step setup guide
- **[docs/CAMERA_SETUP.md](docs/CAMERA_SETUP.md)** - Camera troubleshooting
- **[docs/NETWORK_SETUP.md](docs/NETWORK_SETUP.md)** - Network configuration
- **[docs/WHEEL_TROUBLESHOOTING.md](docs/WHEEL_TROUBLESHOOTING.md)** - Motor diagnostics
- **[docs/README_MANUAL_CONTROL.md](docs/README_MANUAL_CONTROL.md)** - Manual control guide

---

## 📄 License

MIT License
