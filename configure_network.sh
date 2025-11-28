#!/bin/bash
# Script to configure network for TurtleBot communication
# Run with: sudo ./configure_network.sh

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0"
    exit 1
fi

echo "=================================================="
echo "TurtleBot Network Configuration"
echo "=================================================="
echo ""
echo "Current network configuration:"
ip addr show enp0s1 | grep "inet "
echo ""
echo "TurtleBot is on: 192.168.0.18"
echo ""
echo "Choose configuration method:"
echo "1. Add static IP 192.168.0.10 (keeps current network)"
echo "2. Replace with static IP 192.168.0.10"
echo "3. Show current netplan config"
echo "4. Cancel"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Adding secondary IP address to enp0s1..."
        ip addr add 192.168.0.10/24 dev enp0s1
        ip addr show enp0s1 | grep "inet "
        echo ""
        echo "✓ Added 192.168.0.10"
        echo "Note: This change is temporary (lost on reboot)"
        echo "To make permanent, edit /etc/netplan/50-cloud-init.yaml"
        ;;
    2)
        echo ""
        echo "This will create a netplan config to use 192.168.0.10"
        echo "Your current network (192.168.64.x) will be replaced"
        read -p "Continue? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            cat > /etc/netplan/99-turtlebot.yaml << 'EOF'
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
EOF
            echo "Applying netplan configuration..."
            netplan apply
            echo ""
            echo "✓ Network configured"
            ip addr show enp0s1 | grep "inet "
        fi
        ;;
    3)
        echo ""
        echo "Current netplan configuration:"
        cat /etc/netplan/*.yaml
        ;;
    4)
        echo "Cancelled"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Testing connection to TurtleBot..."
if ping -c 2 192.168.0.18 > /dev/null 2>&1; then
    echo "✓ Can reach TurtleBot at 192.168.0.18"
else
    echo "✗ Cannot reach TurtleBot at 192.168.0.18"
    echo "  Check that TurtleBot is powered on and on the same network"
fi
