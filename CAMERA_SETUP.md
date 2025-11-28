# Safe Camera Setup Guide

## Step-by-Step Camera Enablement

### 1. Copy the script to TurtleBot

From your **host computer**:
```bash
scp /home/turtlebot/vision360/enable_camera_safe.sh ubuntu@192.168.0.18:~/
```

### 2. SSH into TurtleBot

```bash
ssh ubuntu@192.168.0.18
```

### 3. Run the safe enablement script

```bash
cd ~
chmod +x enable_camera_safe.sh
./enable_camera_safe.sh
```

The script will:
- ✅ Create automatic backup of config
- ✅ Show you current settings
- ✅ Ask before making changes
- ✅ Provide rollback instructions
- ✅ Offer to reboot for you

### 4. After Reboot - Check Camera

SSH back in and check:
```bash
v4l2-ctl --list-devices
```

You should see something like:
```
bcm2835-isp (platform:bcm2835-isp):
    /dev/video0
    /dev/video1
    ...
```

Or for Raspberry Pi Camera:
```
mmal service 16.1 (platform:bcm2835-v4l2):
    /dev/video0
```

### 5. Test Camera Capture

Simple test:
```bash
# Using v4l2
v4l2-ctl --device=/dev/video0 --set-fmt-video=width=640,height=480,pixelformat=MJPG
v4l2-ctl --device=/dev/video0 --stream-mmap --stream-count=1 --stream-to=/tmp/test.jpg

# Check if image was captured
ls -lh /tmp/test.jpg
```

### 6. Start Camera Streaming

Once camera works:
```bash
cd ~/vision360
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
python3 fix_camera_streamer.py
```

## Rollback if Something Goes Wrong

If the system doesn't boot or has issues:

### Option 1: Use the backup
```bash
# Find your backup
ls -la /boot/firmware/config.txt.backup_*

# Restore it
sudo cp /boot/firmware/config.txt.backup_YYYYMMDD_HHMMSS /boot/firmware/config.txt
sudo reboot
```

### Option 2: Manual fix
```bash
sudo nano /boot/firmware/config.txt

# Change camera_auto_detect=1 back to 0
# Or comment it out with #

# Save and reboot
sudo reboot
```

## Common Issues & Solutions

### Camera not detected after reboot

**Check connection:**
- Power off TurtleBot completely
- Check camera ribbon cable is firmly seated
- Try reconnecting the camera
- Power back on

**Check camera hardware:**
```bash
# On Raspberry Pi 4/5
vcgencmd get_camera

# Should show: supported=1 detected=1
```

### Wrong /dev/video device

The camera might be on a different video device. Try:
```bash
# Test each device
for dev in /dev/video{0..3}; do
    echo "Testing $dev"
    v4l2-ctl --device=$dev --info 2>/dev/null | head -5
done
```

### USB Camera (when it arrives)

Good news - USB cameras are often easier!

Just plug it in and it should appear as `/dev/video0` (or similar).

Test:
```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```

The `fix_camera_streamer.py` script will automatically find it!

## When USB Camera Arrives

1. Plug USB camera into TurtleBot
2. Check: `ls /dev/video*`
3. Run: `python3 fix_camera_streamer.py`
4. It should auto-detect the USB camera!

USB cameras typically show up as:
```
USB Camera (usb-0000:01:00.0):
    /dev/video0
    /dev/video1
```

Much simpler than Pi Camera!
