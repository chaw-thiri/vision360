# Vision360 Web Interface

Web-based control and fleet management interface for TurtleBot autonomous driving system.

## Features

### Driver Interface
- **Mode Selection**: Switch between Manual and Autonomous modes
- **Manual Control**: Web-based driving controls (WASD/Arrow keys or on-screen buttons)
- **Live Camera Feed**: Real-time video stream from TurtleBot
- **Status Monitoring**: Real-time display of robot state, velocities, detections, and FPS

### Manager Interface
- **Fleet Dashboard**: Monitor multiple vehicles (1 real + simulated)
- **Vehicle Management**: Add, edit, delete vehicles (CRUD operations)
- **Real-time Updates**: Live tracking of battery, location, and status
- **Korean Locations**: Realistic South Korean warehouse/distribution locations

## Quick Start

### 1. Install Dependencies
```bash
cd /home/turtlebot/vision360/web
pip3 install -r requirements-web.txt
```

### 2. Start the Web Interface
```bash
./start_web_interface.sh
```

### 3. Access the Interface
Open your browser and navigate to:
```
http://192.168.0.18:8000
```
(Replace with your TurtleBot's IP address)

## Architecture

```
Browser ←→ FastAPI (WebSocket + REST) ←→ ROS Bridge ←→ ROS 2 Topics ←→ TurtleBot
```

## File Structure

```
web/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── ros_bridge.py        # ROS 2 integration
│   ├── websocket_manager.py # WebSocket handling
│   ├── mock_data.py         # Fleet simulation
│   └── models.py            # Data models
├── frontend/
│   ├── index.html           # Role selection page
│   ├── driver.html          # Driver interface
│   ├── manager.html         # Manager dashboard
│   ├── css/styles.css       # Blue accent theme
│   └── js/
│       ├── common.js        # WebSocket client
│       ├── driver.js        # Driver controls
│       └── manager.js       # Fleet management
├── requirements-web.txt
└── start_web_interface.sh
```

## Usage

### Driver Controls

**Keyboard (Manual Mode)**:
- `W` / `↑` : Forward
- `S` / `↓` : Backward
- `A` / `←` : Turn Left
- `D` / `→` : Turn Right

**On-Screen Buttons**: Click and hold directional buttons

**Emergency Stop**: Red button stops robot immediately

### Manager Operations

**Add Vehicle**: Click "+ Add Vehicle" button
**Edit Vehicle**: Click "Edit" on any simulated vehicle card
**Delete Vehicle**: Click "Delete" (real robot TB3-001 cannot be deleted)

## Configuration

The system uses realistic South Korean locations:
- Incheon Airport Warehouse
- Gangnam Distribution Hub
- Songdo Smart City Depot
- And more...

Mock vehicles display:
- Vehicle ID, Model
- Product tag
- Journey (Departure → Destination)
- Current location (lat/lon)
- Battery status with visual indicator
- Status badge (Active/Idle/Charging)

## Notes

- Real robot (TB3-001) marked with ⚡ badge
- Simulated vehicles for demo purposes
- Blue accent theme (#2563eb) throughout
- Responsive design for desktop and mobile
- Auto-reconnecting WebSocket

## Troubleshooting

**No camera feed**: Ensure vision_node is running and publishing to `/camera/image_raw/compressed`

**Controls not working**: Check mode is set to "Manual" and WebSocket is connected

**Connection issues**: Verify TurtleBot and computer are on same network (192.168.0.x)
