#!/bin/bash
# Safe Raspberry Pi Camera Enablement Script
# Creates backups and allows easy rollback

set -e  # Exit on error

echo "=========================================="
echo "Safe Raspberry Pi Camera Enablement"
echo "=========================================="
echo ""

# Check if running on TurtleBot
if [ ! -f "/boot/firmware/config.txt" ]; then
    echo "❌ Error: /boot/firmware/config.txt not found"
    echo "   This script should run on the TurtleBot (Raspberry Pi)"
    exit 1
fi

echo "Step 1: Creating backup..."
BACKUP_FILE="/boot/firmware/config.txt.backup_$(date +%Y%m%d_%H%M%S)"
sudo cp /boot/firmware/config.txt "$BACKUP_FILE"
echo "✓ Backup created: $BACKUP_FILE"
echo ""

echo "Step 2: Current camera configuration:"
grep -i camera /boot/firmware/config.txt || echo "  (No camera config found)"
echo ""

echo "Step 3: Checking what needs to be changed..."
if grep -q "^camera_auto_detect=1" /boot/firmware/config.txt; then
    echo "✓ Camera is already enabled!"
    echo ""
    echo "Let's check if camera is detected..."
    v4l2-ctl --list-devices 2>/dev/null || echo "  No camera devices found"
    exit 0
elif grep -q "^camera_auto_detect=0" /boot/firmware/config.txt; then
    echo "⚠ Camera is disabled (camera_auto_detect=0)"
    echo ""
    read -p "Enable camera? (yes/no): " response
    if [ "$response" = "yes" ]; then
        echo ""
        echo "Enabling camera..."
        sudo sed -i 's/^camera_auto_detect=0/camera_auto_detect=1/' /boot/firmware/config.txt
        echo "✓ Camera enabled in config"
    else
        echo "Cancelled"
        exit 0
    fi
elif grep -q "^#camera_auto_detect" /boot/firmware/config.txt; then
    echo "⚠ Camera setting is commented out"
    echo ""
    read -p "Uncomment and enable camera? (yes/no): " response
    if [ "$response" = "yes" ]; then
        echo ""
        echo "Enabling camera..."
        sudo sed -i 's/^#camera_auto_detect=.*/camera_auto_detect=1/' /boot/firmware/config.txt
        echo "✓ Camera enabled in config"
    else
        echo "Cancelled"
        exit 0
    fi
else
    echo "⚠ No camera_auto_detect setting found"
    echo ""
    read -p "Add camera_auto_detect=1 to config? (yes/no): " response
    if [ "$response" = "yes" ]; then
        echo ""
        echo "Adding camera configuration..."
        echo "camera_auto_detect=1" | sudo tee -a /boot/firmware/config.txt
        echo "✓ Camera setting added"
    else
        echo "Cancelled"
        exit 0
    fi
fi

echo ""
echo "Step 4: Verifying changes..."
echo "New camera configuration:"
grep -i camera /boot/firmware/config.txt
echo ""

echo "=========================================="
echo "✓ Configuration updated successfully!"
echo "=========================================="
echo ""
echo "⚠ REBOOT REQUIRED for changes to take effect"
echo ""
echo "Next steps:"
echo "1. Reboot: sudo reboot"
echo "2. After reboot, check: v4l2-ctl --list-devices"
echo "3. Look for 'bcm2835-isp' or camera device"
echo ""
echo "To rollback if needed:"
echo "  sudo cp $BACKUP_FILE /boot/firmware/config.txt"
echo "  sudo reboot"
echo ""
read -p "Reboot now? (yes/no): " reboot_response

if [ "$reboot_response" = "yes" ]; then
    echo ""
    echo "Rebooting in 5 seconds... (Ctrl+C to cancel)"
    sleep 5
    sudo reboot
else
    echo ""
    echo "Remember to reboot later: sudo reboot"
fi
