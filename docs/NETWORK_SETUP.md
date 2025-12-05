# Network Configuration Guide

Your host computer is on `192.168.64.3` and TurtleBot is on `192.168.0.18`. They need to be on the same subnet for ROS 2 communication.

## Quick Option: Add Secondary IP (Temporary)

This adds a second IP address without affecting your current network:

```bash
sudo ip addr add 192.168.0.10/24 dev enp0s1
```

**Verify:**
```bash
ip addr show enp0s1 | grep "inet "
# Should show both 192.168.64.3 and 192.168.0.10
```

**Test connection:**
```bash
ping 192.168.0.18
```

**Note:** This is temporary and will be lost on reboot.

---

## Using the Configuration Script

I created an interactive script:

```bash
cd /home/turtlebot/vision360
sudo ./configure_network.sh
```

Choose option 1 to add a secondary IP (recommended).

---

## Manual Configuration Options

### Option A: Netplan (Permanent Static IP)

1. **View current config:**
   ```bash
   sudo cat /etc/netplan/50-cloud-init.yaml
   ```

2. **Create new config:**
   ```bash
   sudo nano /etc/netplan/99-turtlebot.yaml
   ```

3. **Add this content:**
   ```yaml
   network:
     version: 2
     renderer: networkd
     ethernets:
       enp0s1:
         dhcp4: no
         addresses:
           - 192.168.0.10/24
         routes:
           - to: default
             via: 192.168.0.1
         nameservers:
           addresses: [8.8.8.8, 8.8.4.4]
   ```

4. **Apply:**
   ```bash
   sudo netplan apply
   ```

### Option B: Keep Both Networks (Multi-IP)

Add to your netplan config:
```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp0s1:
      dhcp4: yes
      addresses:
        - 192.168.0.10/24
```

This keeps DHCP for 192.168.64.x and adds static 192.168.0.10.

---

## After Network Configuration

Once on the same network, test ROS 2 connection:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=17
export ROS_LOCALHOST_ONLY=0

# Check if TurtleBot topics appear
ros2 topic list

# Should see /cmd_vel, /imu, /scan, etc.
```

Then run the movement test:

```bash
cd /home/turtlebot/vision360
./start_movement_test.sh
```

---

## Troubleshooting

### Can't ping TurtleBot

- Check TurtleBot is powered on
- Verify router/switch connects both devices
- Check firewall: `sudo ufw status`

### Topics still not visible

- Verify ROS_DOMAIN_ID=17 on both devices
- Check: `echo $ROS_DOMAIN_ID`
- Restart TurtleBot's robot.launch.py

### Need to revert network

Remove the config:
```bash
sudo rm /etc/netplan/99-turtlebot.yaml
sudo netplan apply
```

Or remove the temporary IP:
```bash
sudo ip addr del 192.168.0.10/24 dev enp0s1
```
