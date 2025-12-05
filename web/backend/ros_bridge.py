"""
ROS 2 Bridge for Vision360 Web Interface
Handles communication between FastAPI web server and ROS 2 topics
"""
import threading
import asyncio
import base64
import time
from typing import Optional, Dict
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import CompressedImage
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS 2 not available")


class ROSBridge(Node):
    """ROS 2 bridge node for web interface"""

    def __init__(self, camera_queue: asyncio.Queue, status_queue: asyncio.Queue):
        super().__init__('vision360_web_bridge')

        self.camera_queue = camera_queue
        self.status_queue = status_queue

        # Latest data cache
        self.latest_status = {}
        self.latest_frame = None
        self.last_camera_time = 0

        # QoS profile for camera (same as vision_node)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to camera feed
        self.camera_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_callback,
            qos_profile
        )

        # Subscribe to autonomous status
        self.status_sub = self.create_subscription(
            String,
            '/autonomous_driving/status',
            self.status_callback,
            10
        )

        # Subscribe to mode control (for future vision_node integration)
        self.mode_sub = self.create_subscription(
            String,
            '/web/control_mode',
            self.mode_callback,
            10
        )

        # Publish manual control commands
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Publish mode control
        self.mode_pub = self.create_publisher(
            String,
            '/web/control_mode',
            10
        )

        # Control state
        self.manual_mode_active = False
        self.last_control_time = 0
        self.control_timeout = 0.5  # 500ms timeout for manual control

        # Frame rate limiting (max 20 FPS for web)
        self.min_frame_interval = 1.0 / 20

        self.get_logger().info("ROS Bridge initialized")
        self.get_logger().info("Subscribing to /camera/image_raw/compressed")
        self.get_logger().info("Subscribing to /autonomous_driving/status")
        self.get_logger().info("Publishing to /cmd_vel")

    def camera_callback(self, msg: CompressedImage):
        """Handle incoming camera frames"""
        current_time = time.time()

        # Rate limit frames
        if current_time - self.last_camera_time < self.min_frame_interval:
            return

        self.last_camera_time = current_time

        try:
            # Convert compressed image to base64
            image_base64 = base64.b64encode(msg.data).decode('utf-8')
            frame_data = {
                'frame': f'data:image/jpeg;base64,{image_base64}',
                'timestamp': current_time
            }

            # Put in async queue (non-blocking)
            try:
                self.camera_queue.put_nowait(frame_data)
            except asyncio.QueueFull:
                # Skip frame if queue is full
                pass

            self.latest_frame = frame_data

        except Exception as e:
            self.get_logger().error(f"Error processing camera frame: {e}")

    def status_callback(self, msg: String):
        """Handle autonomous driving status updates"""
        try:
            # Parse pipe-delimited status string
            # Format: "state:moving|linear:0.15|angular:-0.2|pedestrian:none|..."
            status_dict = {}
            for pair in msg.data.split('|'):
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    status_dict[key] = value

            # Convert numeric values
            if 'linear' in status_dict:
                status_dict['linear'] = float(status_dict['linear'])
            if 'angular' in status_dict:
                status_dict['angular'] = float(status_dict['angular'])
            if 'fps' in status_dict:
                status_dict['fps'] = int(float(status_dict['fps']))

            # Put in async queue
            try:
                self.status_queue.put_nowait(status_dict)
            except asyncio.QueueFull:
                # Skip if queue is full
                pass

            self.latest_status = status_dict

        except Exception as e:
            self.get_logger().error(f"Error parsing status: {e}")

    def mode_callback(self, msg: String):
        """Handle mode control messages"""
        mode = msg.data.lower()
        self.get_logger().info(f"Mode changed to: {mode}")
        self.manual_mode_active = (mode == "manual")

    def publish_manual_control(self, linear: float, angular: float):
        """Publish manual control command to /cmd_vel"""
        # Clamp velocities to safe limits
        linear = max(-0.22, min(0.22, linear))
        angular = max(-2.84, min(2.84, angular))

        twist = Twist()
        twist.linear.x = float(linear)
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = float(angular)

        self.cmd_vel_pub.publish(twist)
        self.last_control_time = time.time()

        # Log occasionally (not every command)
        if abs(linear) > 0.01 or abs(angular) > 0.01:
            self.get_logger().debug(f"Manual control: linear={linear:.2f}, angular={angular:.2f}")

    def publish_mode(self, mode: str):
        """Publish mode control message"""
        msg = String()
        msg.data = mode.lower()
        self.mode_pub.publish(msg)
        self.get_logger().info(f"Published mode: {mode}")

    def check_control_timeout(self):
        """Check if manual control has timed out (safety feature)"""
        if self.manual_mode_active:
            if time.time() - self.last_control_time > self.control_timeout:
                # No commands received recently, send stop
                self.publish_manual_control(0.0, 0.0)

    def get_latest_status(self) -> Dict:
        """Get latest cached status"""
        return self.latest_status.copy()

    def get_latest_frame(self) -> Optional[Dict]:
        """Get latest cached frame"""
        return self.latest_frame


class ROSBridgeThread:
    """Wrapper to run ROS bridge in separate thread"""

    def __init__(self):
        self.camera_queue = None
        self.status_queue = None
        self.bridge_node = None
        self.thread = None
        self.running = False

    def start(self, camera_queue: asyncio.Queue, status_queue: asyncio.Queue):
        """Start ROS bridge in separate thread"""
        if not ROS_AVAILABLE:
            print("Error: ROS 2 not available. Cannot start bridge.")
            return False

        self.camera_queue = camera_queue
        self.status_queue = status_queue
        self.running = True

        self.thread = threading.Thread(target=self._run_ros, daemon=True)
        self.thread.start()

        print("ROS Bridge thread started")
        return True

    def _run_ros(self):
        """Run ROS in separate thread"""
        try:
            rclpy.init()
            self.bridge_node = ROSBridge(self.camera_queue, self.status_queue)

            # Spin with timeout check
            while self.running:
                rclpy.spin_once(self.bridge_node, timeout_sec=0.1)
                # Check control timeout periodically
                self.bridge_node.check_control_timeout()

        except Exception as e:
            print(f"Error in ROS bridge thread: {e}")
        finally:
            if self.bridge_node:
                self.bridge_node.destroy_node()
            rclpy.shutdown()

    def stop(self):
        """Stop ROS bridge"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def publish_control(self, linear: float, angular: float):
        """Publish manual control command"""
        if self.bridge_node:
            self.bridge_node.publish_manual_control(linear, angular)

    def publish_mode(self, mode: str):
        """Publish mode control"""
        if self.bridge_node:
            self.bridge_node.publish_mode(mode)

    def get_latest_status(self) -> Dict:
        """Get latest status"""
        if self.bridge_node:
            return self.bridge_node.get_latest_status()
        return {}

    def get_latest_frame(self) -> Optional[Dict]:
        """Get latest frame"""
        if self.bridge_node:
            return self.bridge_node.get_latest_frame()
        return None
