#!/bin/bash
# Automated Vision System Startup - Run on Desktop/VM
# Supports: ROS mode (with TurtleBot), Webcam mode, Video mode

# Parse command line arguments
MODE="${1:-ros}"  # Default to ros mode
INPUT_FILE="$2"

echo "=========================================="
echo "Vision360 Desktop Startup"
echo "=========================================="
echo ""

# Display mode
case "$MODE" in
    ros)
        echo "Mode: ROS (TurtleBot Required)"
        echo ""
        ;;
    webcam)
        echo "Mode: Webcam (Standalone Testing)"
        echo ""
        ;;
    video)
        echo "Mode: Video File (Standalone Testing)"
        if [ -z "$INPUT_FILE" ]; then
            echo "Error: Video mode requires input file"
            echo "Usage: $0 video <path/to/video.mp4>"
            exit 1
        fi
        echo "Input: $INPUT_FILE"
        echo ""
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo ""
        echo "Usage:"
        echo "  $0                          # ROS mode (default)"
        echo "  $0 ros                      # ROS mode with TurtleBot"
        echo "  $0 webcam                   # Webcam mode (no robot needed)"
        echo "  $0 video <path/to/file>     # Video file mode"
        echo ""
        exit 1
        ;;
esac

# ROS mode specific checks
if [ "$MODE" == "ros" ]; then
    # Setup ROS environment
    if [ -f /opt/ros/humble/setup.bash ]; then
        source /opt/ros/humble/setup.bash
        export ROS_DOMAIN_ID=17
        export ROS_LOCALHOST_ONLY=0

        # Check if TurtleBot is ready
        echo "Checking TurtleBot connection..."

        if ! ros2 topic list 2>/dev/null | grep -q "/cmd_vel"; then
            echo "✗ TurtleBot not detected!"
            echo ""
            echo "Make sure the TurtleBot startup script is running:"
            echo "  ssh turtlebot@192.168.0.18"
            echo "  cd ~/vision360"
            echo "  ./scripts/start_turtlebot.sh"
            echo ""
            echo "Or run in standalone mode:"
            echo "  $0 webcam                   # Use webcam"
            echo "  $0 video <file.mp4>         # Use video file"
            echo ""
            exit 1
        fi

        echo "✓ Robot base connected"

        if ! ros2 topic list 2>/dev/null | grep -q "/camera/image_raw/compressed"; then
            echo "⚠ Camera not detected!"
            echo "  Make sure camera streamer is running on TurtleBot"
            echo ""
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            echo "✓ Camera connected"
        fi

        echo ""
        echo "=========================================="
        echo "⚠ SAFETY CHECK"
        echo "=========================================="
        echo "Before starting:"
        echo "  • Robot has 2m clear space"
        echo "  • You can reach keyboard for emergency stop"
        echo "  • Battery is charged"
        echo ""
        read -p "Ready to start? (yes/no): " response

        if [ "$response" != "yes" ]; then
            echo "Cancelled"
            exit 0
        fi
    else
        echo "Warning: ROS 2 Humble not found at /opt/ros/humble"
        echo "Attempting to run anyway..."
    fi
fi

echo ""
echo "Starting Vision360 System..."
echo "Controls:"
if [ "$MODE" == "ros" ]; then
    echo "  Q = Emergency Stop"
    echo "  R = Resume"
    echo "  X = Quit"
    echo "  T = Toggle Debug"
else
    echo "  X = Quit"
    echo "  Space = Pause/Resume (video mode)"
    echo "  T = Toggle Debug"
fi
echo ""
sleep 2

# Navigate to project directory
cd "$(dirname "$0")/.." || exit 1

# Run based on mode
case "$MODE" in
    ros)
        python3 main.py --mode ros
        ;;
    webcam)
        python3 main.py --mode webcam --camera 0
        ;;
    video)
        python3 main.py --mode video --input "$INPUT_FILE"
        ;;
esac
