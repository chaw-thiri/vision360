#!/bin/bash
# Script to identify which video device is the actual camera

echo "=========================================="
echo "Camera Device Identification"
echo "=========================================="
echo ""

echo "1. Listing all video devices:"
ls -l /dev/video* 2>/dev/null
echo ""

echo "2. Video device information:"
for dev in /dev/video{0..31}; do
    if [ -e "$dev" ]; then
        echo "--- $dev ---"
        v4l2-ctl --device=$dev --info 2>/dev/null || echo "  (No v4l2 info available)"
        v4l2-ctl --device=$dev --list-formats 2>/dev/null | head -5
        echo ""
    fi
done

echo "=========================================="
echo "Checking camera access:"
echo "=========================================="

# Check user groups
echo "Current user groups:"
groups
echo ""

# Check video group
echo "Users in 'video' group:"
getent group video
echo ""

echo "=========================================="
echo "Suggested fix if permission denied:"
echo "  sudo usermod -aG video $USER"
echo "  Then logout and login again"
echo "=========================================="
