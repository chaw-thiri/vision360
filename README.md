# Autonomous Driving Vision System for TurtleBot3

A computer vision application for autonomous driving on TurtleBot3 Burger, featuring pedestrian detection, lane detection, and traffic light recognition.

## Features

- **Pedestrian Detection**: YOLOv8-based real-time person detection with danger level assessment
- **Lane Detection**: Color-based detection for indoor taped tracks (white, yellow, blue tape)
- **Traffic Light Detection**: HSV color segmentation for red, yellow, and green lights
- **Boundary Platform Detection**: Black platform detection with zone analysis for obstacle avoidance
- **Manual Control Mode**: Keyboard-based manual override with arrow keys and WASD controls
- **Decision Making**: Priority-based navigation controller with PID lane following
- **ROS 2 Integration**: Full integration with TurtleBot3 via ROS 2 Humble
- **Automated Setup Scripts**: Shell scripts for quick TurtleBot and vision system deployment

## System Architecture

```
┌─────────────────────┐                    ┌─────────────────────────────┐
│  TurtleBot3 Burger  │                    │     Host Computer           │
│   (Raspberry Pi)    │   WiFi Network     │  (Mac/Windows/Linux)        │
│                     │                    │                             │
│  ┌───────────────┐  │  /camera/image     │  ┌─────────────────────┐    │
│  │   Pi Camera   │──┼──────────────────►─┼──│   Person Detector   │    │
│  │    (Wide)     │  │   (Compressed)     │  │      (YOLOv8)       │    │
│  └───────────────┘  │                    │  └─────────┬───────────┘    │
│                     │                    │            │                │
│                     │                    │  ┌─────────▼───────────┐    │
│                     │                    │  │   Lane Detector     │    │
│                     │                    │  │  (HSV + Hough)      │    │
│                     │                    │  └─────────┬───────────┘    │
│                     │                    │            │                │
│                     │                    │  ┌─────────▼───────────┐    │
│  ┌───────────────┐  │    /cmd_vel        │  │ Traffic Light Det.  │    │
│  │   Motors      │◄─┼──────────────────◄─┼──│   (HSV + Contour)   │    │
│  │  (OpenCR)     │  │  (Twist message)   │  └─────────┬───────────┘    │
│  └───────────────┘  │                    │            │                │
│                     │                    │  ┌─────────▼───────────┐    │
└─────────────────────┘                    │  │  Decision Maker     │    │
                                           │  │  (PID Controller)   │    │
                                           │  └─────────────────────┘    │
                                           └─────────────────────────────┘
```

## Algorithms Used

### 1. Person Detection - YOLOv8 (You Only Look Once v8)

**Algorithm**: Deep learning-based object detection using Convolutional Neural Networks (CNN)

**How it works**:
- Uses YOLOv8 nano model (`yolov8n.pt`) optimized for real-time inference
- Single forward pass through the network detects all persons in frame
- Outputs bounding boxes, confidence scores, and class predictions
- Distance estimation based on bounding box height relative to frame size

```
Input Image → CNN Backbone → Feature Pyramid → Detection Head → Bounding Boxes
                (CSPNet)      Network (FPN)     (Decoupled)
```

### 2. Lane Detection - HSV Color Segmentation + Hough Transform

**Algorithms**:
- HSV Color Thresholding
- Canny Edge Detection
- Probabilistic Hough Line Transform

**How it works**:
1. **Color Segmentation**: Convert BGR to HSV color space, apply thresholds to isolate tape colors (white/yellow/blue)
2. **Edge Detection**: Canny algorithm finds edges using gradient magnitude and non-maximum suppression
3. **Line Detection**: Hough Transform converts edge points to parameter space (ρ, θ) to find line candidates
4. **Lane Separation**: Lines are classified as left/right based on slope and position
5. **Lane Averaging**: Multiple line segments are averaged using polynomial fitting

```
Frame → HSV Convert → Color Mask → Canny Edges → Hough Lines → Left/Right Separation → Average Lane
              ↓
        Region of Interest (ROI) - Bottom 50% of frame
```

### 3. Traffic Light Detection - HSV Segmentation + Contour Analysis

**Algorithms**:
- HSV Color Space Segmentation
- Morphological Operations (Opening/Closing)
- Contour Detection and Circularity Analysis

**How it works**:
1. **ROI Extraction**: Focus on upper portion of frame where traffic lights appear
2. **Color Isolation**: Separate masks for red, yellow, green using HSV ranges
3. **Noise Removal**: Morphological opening removes small noise, closing fills gaps
4. **Shape Detection**: Find contours and filter by circularity (4πA/P²) and area
5. **State Smoothing**: Temporal filtering over 5 frames for stable detection

```
Frame → ROI Crop → HSV → Color Masks (R/Y/G) → Morphology → Contours → Circularity Filter → State
```

### 4. Boundary Platform Detection - HSV Segmentation + Zone Analysis

**Algorithms**:
- HSV Color Space Thresholding for black platform detection
- Morphological Operations (Opening/Closing)
- Zone-based coverage analysis (left/center/right)

**How it works**:
1. **ROI Extraction**: Focus on bottom 40% of frame where boundary platforms appear
2. **Black Detection**: HSV thresholding to isolate black platforms (V < 50)
3. **Noise Removal**: Morphological closing fills gaps, opening removes small noise
4. **Zone Analysis**: Divide frame into left (0-33%), center (33-67%), right (67-100%)
5. **Coverage Calculation**: Measure percentage of each zone covered by black platforms
6. **Navigation Decision**:
   - Block threshold: 30% coverage = zone blocked
   - Suggests avoidance direction based on open zones
   - Calculates avoidance angle for smooth steering adjustment

```
Frame → ROI (bottom 40%) → HSV → Black Mask → Morphology → Zone Division → Coverage Analysis → Avoidance Angle
                                                                  ↓
                                                    Left | Center | Right zones
```

### 5. Decision Making - Priority-Based Controller with PID

**Algorithms**:
- Priority Queue Decision System
- PID (Proportional-Integral-Derivative) Controller for lane following
- Velocity Smoothing with acceleration limits

**How it works**:
1. **Priority System**: Pedestrian safety (P1) > Boundary platforms (P2) > Traffic lights (P3) > Lane following (P4)
2. **PID Lane Following**:
   - Error = lane center offset from frame center
   - Output = Kp×error + Ki×∫error + Kd×(d/dt)error
3. **Velocity Smoothing**: Limits acceleration/deceleration for smooth motion

```
                    ┌─────────────┐
Pedestrian Status ──►│             │
                    │  Priority   │──► Linear Velocity
Boundary Platforms ─►│  Decision  │
                    │   Maker    │──► Angular Velocity
Traffic Light ──────►│             │
                    │             │
Lane Offset ────────►│             │
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  Velocity   │
                    │ Smoother    │──► /cmd_vel
                    └─────────────┘
```

## How The System Works (Process Flow)

### Step 1: Image Acquisition
The Raspberry Pi Camera captures frames at 30 FPS and compresses them to JPEG format. These compressed images are published to the `/camera/image_raw/compressed` ROS 2 topic and transmitted over WiFi to the host computer.

### Step 2: Parallel Detection
Upon receiving a frame, four detection modules run:

1. **Person Detector**: YOLOv8 processes the full frame, identifying any humans. For each detection, it calculates a relative distance estimate based on how much of the frame the person occupies. A person taking up 40%+ of the frame triggers an emergency stop.

2. **Lane Detector**: The bottom 50% of the frame (road area) is analyzed. HSV thresholding isolates the tape color, Canny finds edges, and Hough Transform detects line segments. Lines are separated into left/right lanes based on their slope, then averaged to produce stable lane boundaries. The center offset between the lane center and frame center is calculated.

3. **Traffic Light Detector**: The upper 50% of the frame is scanned for circular colored regions. Red/Yellow/Green masks are created, contours extracted, and filtered by circularity. The detected state is smoothed over multiple frames to prevent flickering.

4. **Boundary Platform Detector**: The bottom 40% of the frame is analyzed for black boundary platforms. HSV thresholding isolates black regions, morphological operations clean up noise, and the frame is divided into three zones (left/center/right). Coverage percentage is calculated for each zone to determine which directions are blocked and suggest an optimal avoidance angle.

### Step 3: Decision Making
The Decision Maker receives all detection results and applies priority-based logic:

```
IF pedestrian_too_close:
    → EMERGENCY STOP
ELIF pedestrian_in_path:
    → SLOW DOWN + AVOID (turn away from pedestrian)
ELIF all_zones_blocked_by_boundary:
    → STOP (cannot proceed safely)
ELIF boundary_platform_detected:
    → ADJUST STEERING to avoid blocked zones
ELIF red_light:
    → STOP
ELIF yellow_light:
    → SLOW DOWN
ELSE:
    → FOLLOW LANE using PID controller
```

The PID controller continuously adjusts angular velocity to minimize lane center offset, keeping the robot centered in its lane. When boundary platforms are detected, the avoidance angle is blended with the lane-following command to navigate around obstacles while staying on track.

### Step 4: Velocity Control
Raw velocity commands are smoothed to prevent jerky motion:
- Acceleration is limited to 0.5 m/s² (linear) and 2.0 rad/s² (angular)
- Deceleration is faster (1.0 m/s²) for safety
- Emergency stops use double deceleration rate

### Step 5: Robot Control
The final velocity command (Twist message) is published to `/cmd_vel`. The TurtleBot's OpenCR controller receives this and drives the motors accordingly. The cycle repeats at ~30 Hz.

## Project Structure

```
vision360/
├── config/
│   ├── config.yaml                    # Configuration parameters
│   └── fastdds_peer_config.xml       # FastDDS peer discovery config
├── src/
│   ├── detectors/
│   │   ├── person_detector.py             # YOLOv8 pedestrian detection
│   │   ├── lane_detector.py               # Lane line detection
│   │   ├── traffic_light_detector.py      # Traffic light detection
│   │   └── boundary_platform_detector.py  # Black platform detection
│   ├── controller/
│   │   ├── decision_maker.py              # Navigation decision logic
│   │   └── velocity_controller.py         # Velocity smoothing
│   └── ros_nodes/
│       └── vision_node.py                 # ROS 2 vision node
├── Shell Scripts (Setup & Control)/
│   ├── start_turtlebot.sh            # Launch TurtleBot vision system
│   ├── start_vision_desktop.sh       # Launch vision on desktop
│   ├── start_manual_control.sh       # Start manual keyboard control
│   ├── start_movement_test.sh        # Test robot movement
│   ├── configure_network.sh          # Network setup automation
│   ├── enable_camera_safe.sh         # Safe camera initialization
│   ├── identify_camera.sh            # Identify connected cameras
│   ├── check_robot_status.sh         # Check robot health
│   └── debug_robot.sh                # Debugging utilities
├── Test Scripts/
│   ├── test_movement.py              # Test wheel motors
│   ├── test_wheels.py                # Wheel diagnostics
│   ├── test_vision_with_robot.py     # Integrated vision testing
│   ├── manual_control.py             # Keyboard control interface
│   ├── view_camera.py                # Camera feed viewer
│   └── fix_camera_streamer.py        # Camera troubleshooting
├── Documentation/
│   ├── README.md                     # Main documentation
│   ├── QUICK_START.md                # Quick start guide
│   ├── README_MANUAL_CONTROL.md      # Manual control guide
│   ├── CAMERA_SETUP.md               # Camera configuration
│   ├── NETWORK_SETUP.md              # Network configuration
│   └── WHEEL_TROUBLESHOOTING.md      # Wheel debugging guide
├── main.py                           # Main entry point
└── requirements.txt                  # Python dependencies
```

## Quick Start

For experienced users, automated scripts are available:

**TurtleBot (on robot)**:
```bash
./start_turtlebot.sh
```

**Desktop/Laptop (vision processing)**:
```bash
./start_vision_desktop.sh
```

**Manual Control Mode**:
```bash
./start_manual_control.sh
```

See `QUICK_START.md` for detailed quick start instructions.

## Setup Instructions

### TurtleBot3 Setup (Raspberry Pi)

SSH into your TurtleBot:
```bash
ssh ubuntu@192.168.0.18
```

**Option 1: Using automated script**
```bash
./configure_network.sh
./start_turtlebot.sh
```

**Option 2: Manual setup**

Copy the turtlebot_setup folder to the TurtleBot:
```bash
scp -r turtlebot_setup ubuntu@192.168.0.18:~/capstone/
```

Run the setup script:
```bash
cd ~/capstone/turtlebot_setup
chmod +x setup_turtlebot.sh
./setup_turtlebot.sh
```

Start the camera streamer:
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
python3 camera_streamer.py
```

---

### Host Computer Setup

#### macOS Setup

**Option 1: Using setup script**
```bash
chmod +x scripts/setup_ros2_mac.sh
./scripts/setup_ros2_mac.sh
```

**Option 2: Manual installation**
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Create virtual environment
python3.11 -m venv ~/.venvs/ros2_vision
source ~/.venvs/ros2_vision/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**ROS 2 on macOS** (for full TurtleBot integration):
```bash
# Using Docker (recommended)
docker run -it --rm --net=host -e ROS_DOMAIN_ID=17 \
    -v $(pwd):/workspace ros:humble bash

# Inside container
cd /workspace
pip install -r requirements.txt
python main.py --mode ros
```

---

#### Windows Setup

**Step 1: Install Python**
- Download Python 3.11 from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"

**Step 2: Install dependencies**
```powershell
# Open PowerShell as Administrator

# Create virtual environment
python -m venv C:\venvs\ros2_vision
C:\venvs\ros2_vision\Scripts\Activate.ps1

# Install PyTorch (CPU version, or visit pytorch.org for GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install opencv-python>=4.8.0 numpy>=1.24.0 ultralytics>=8.0.0 PyYAML>=6.0 scipy>=1.11.0 Pillow>=10.0.0

# Download YOLO model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**Step 3: Run standalone mode**
```powershell
# Activate environment
C:\venvs\ros2_vision\Scripts\Activate.ps1

# Run with webcam
python main.py --mode webcam

# Run with video
python main.py --mode video --input path\to\video.mp4
```

**ROS 2 on Windows** (for full TurtleBot integration):
```powershell
# Option A: WSL2 with Ubuntu 22.04 (Recommended)
wsl --install -d Ubuntu-22.04

# Inside WSL2, follow Linux instructions below

# Option B: Docker Desktop
docker run -it --rm --net=host -e ROS_DOMAIN_ID=17 -v ${PWD}:/workspace ros:humble bash
```

---

#### Linux (Ubuntu 22.04) Setup

**Step 1: Install system dependencies**
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-opencv
```

**Step 2: Create virtual environment and install packages**
```bash
# Create virtual environment
python3 -m venv ~/.venvs/ros2_vision
source ~/.venvs/ros2_vision/bin/activate

# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or for NVIDIA GPU:
# pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt

# Download YOLO model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**Step 3: Install ROS 2 Humble (for TurtleBot integration)**
```bash
# Add ROS 2 repository
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2
sudo apt update
sudo apt install -y ros-humble-desktop ros-humble-cv-bridge ros-humble-image-transport

# Source ROS 2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Set domain ID
echo "export ROS_DOMAIN_ID=17" >> ~/.bashrc
source ~/.bashrc
```

**Step 4: Run the application**
```bash
# Standalone mode (webcam)
source ~/.venvs/ros2_vision/bin/activate
python main.py --mode webcam

# ROS 2 mode (with TurtleBot)
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
python main.py --mode ros
```

---

### Network Configuration

Ensure both devices are on the same network:
```bash
# On host computer
export ROS_DOMAIN_ID=17
ping 192.168.0.18

# Verify ROS 2 communication
ros2 topic list  # Should see /camera/image_raw/compressed
```

## Usage

### Standalone Mode (Testing without TurtleBot)

```bash
# Test with webcam
python main.py --mode webcam

# Test with video file
python main.py --mode video --input path/to/video.mp4

# Test with image
python main.py --mode image --input path/to/image.jpg
```

### ROS 2 Mode (Full TurtleBot Integration)

**Terminal 1 - TurtleBot (SSH)**:
```bash
ssh ubuntu@192.168.0.18
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
python3 ~/capstone/turtlebot_setup/camera_streamer.py
```

**Terminal 2 - Host Computer**:
```bash
source ~/.venvs/ros2_vision/bin/activate  # or your venv path
export ROS_DOMAIN_ID=17
python main.py --mode ros
```

### Keyboard Controls

#### Autonomous Mode Controls

| Key | Action |
|-----|--------|
| `Q` | Emergency stop |
| `R` | Resume autonomous control |
| `T` | Toggle debug view |
| `X` | Quit application |
| `Space` | Pause/Resume (video mode) or Resume autonomous |

#### Manual Control Mode

| Key | Action |
|-----|--------|
| `↑` or `W` | Move forward |
| `↓` or `S` | Move backward |
| `←` or `A` | Turn left |
| `→` or `D` | Turn right |
| `Space` | Stop and resume autonomous |
| `+` / `=` | Increase manual speed |
| `-` / `_` | Decrease manual speed |
| `Q` | Emergency stop |
| `R` | Resume autonomous |
| `X` | Quit |

See `README_MANUAL_CONTROL.md` for detailed manual control instructions.

## Configuration

Edit `config/config.yaml` to adjust parameters:

### Key Parameters

```yaml
# Person detection sensitivity
person_detection:
  confidence_threshold: 0.5    # Lower = more detections
  danger_zone_ratio: 0.4       # Person size that triggers stop

# Lane colors (adjust for your track)
lane_detection:
  white_tape:
    v_low: 200    # Increase if detecting too much

# Boundary platform detection
boundary_detection:
  enabled: true
  block_threshold: 0.3         # 30% coverage = zone blocked
  avoidance_strength: 0.6      # How aggressively to avoid (0-1)
  black_platform:
    v_high: 50                 # Max value for black detection

# Robot speeds
turtlebot:
  normal_speed: 0.15    # m/s during normal operation
  slow_speed: 0.08      # m/s when caution
```

## Troubleshooting

### Camera not working on TurtleBot
```bash
./identify_camera.sh      # Identify connected cameras
./enable_camera_safe.sh   # Safe camera initialization
libcamera-hello           # Test camera
```
See `CAMERA_SETUP.md` for detailed camera troubleshooting.

### Wheel/Motor issues
```bash
./test_wheels.py          # Test individual wheels
./test_movement.py        # Test basic movements
./check_robot_status.sh   # Check robot health
```
See `WHEEL_TROUBLESHOOTING.md` for detailed wheel diagnostics.

### ROS 2 communication issues
```bash
./configure_network.sh    # Automated network setup
echo $ROS_DOMAIN_ID       # Must be 17 on both devices
ros2 topic list           # Check if topics are visible
```
See `NETWORK_SETUP.md` for detailed network configuration.

### Poor lane detection
- Adjust HSV values in config for your tape color
- Ensure adequate, consistent lighting
- Verify ROI covers the road area
- Use debug mode (`T` key) to visualize detection

### Boundary platform not detecting
- Ensure platforms are dark/black (V < 50 in HSV)
- Adjust `block_threshold` in config
- Check ROI covers bottom 40% where platforms appear
- Verify lighting isn't creating harsh shadows

### YOLO runs slowly
- Use GPU if available (install CUDA + GPU PyTorch)
- Reduce frame resolution in config
- Use `yolov8n.pt` (nano) for fastest inference

### Robot not responding to commands
```bash
./debug_robot.sh          # Run diagnostics
./check_robot_status.sh   # Check status
```

## Testing & Diagnostics

The system includes comprehensive testing and diagnostic tools:

### Vision Testing
```bash
# Test vision system with robot
python test_vision_with_robot.py

# View live camera feed
python view_camera.py

# Test with video file
python main.py --mode video --input test_video.mp4
```

### Movement Testing
```bash
# Test basic movements
./start_movement_test.sh
python test_movement.py

# Test individual wheels
python test_wheels.py

# Manual control testing
./start_manual_control.sh
python manual_control.py
```

### System Diagnostics
```bash
# Check overall robot status
./check_robot_status.sh

# Run full diagnostics
./debug_robot.sh

# Fix camera issues
python fix_camera_streamer.py
```

## Additional Documentation

- **QUICK_START.md** - Quick start guide for fast deployment
- **README_MANUAL_CONTROL.md** - Detailed manual control instructions
- **CAMERA_SETUP.md** - Camera configuration and troubleshooting
- **NETWORK_SETUP.md** - Network configuration guide
- **WHEEL_TROUBLESHOOTING.md** - Wheel and motor diagnostics

## Dependencies

- Python 3.10+
- OpenCV >= 4.8.0
- PyTorch >= 2.0.0
- Ultralytics (YOLOv8) >= 8.0.0
- ROS 2 Humble (optional, for TurtleBot integration)
- NumPy, SciPy, PyYAML

## License

MIT License
