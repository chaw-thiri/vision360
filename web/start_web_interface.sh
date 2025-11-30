#!/bin/bash
# Start Vision360 Web Interface

cd "$(dirname "$0")/backend"

echo "Starting Vision360 Web Interface..."
echo "Access at: http://$(hostname -I | awk '{print $1}'):8000"

# Activate ROS 2 environment if available
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

# Install dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip3 install -r ../requirements-web.txt
fi

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
