#!/usr/bin/env python3
"""
Camera Streamer for TurtleBot3 (Raspberry Pi).
Publishes camera images to ROS 2 topics for remote processing.

Run this script on the TurtleBot's Raspberry Pi.
"""

import sys
import time

# ROS 2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import CompressedImage, CameraInfo
    from std_msgs.msg import Header
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Error: ROS 2 not available")
    sys.exit(1)

# Camera imports
try:
    from picamera2 import Picamera2
    import cv2
    import numpy as np
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Warning: picamera2 not available, using OpenCV")
    import cv2
    import numpy as np


class CameraStreamerNode(Node):
    """ROS 2 node for streaming Raspberry Pi camera."""

    def __init__(self):
        super().__init__('camera_streamer_node')

        # Declare parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('jpeg_quality', 80)

        # Get parameters
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value

        # QoS profile for image publishing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.image_pub = self.create_publisher(
            CompressedImage,
            '/camera/image_raw/compressed',
            qos_profile
        )

        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            '/camera/camera_info',
            qos_profile
        )

        # Initialize camera
        self.camera = None
        self._init_camera()

        # Create timer for publishing
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Frame counter for stats
        self.frame_count = 0
        self.start_time = time.time()

        self.get_logger().info(
            f"Camera streamer initialized: {self.width}x{self.height} @ {self.fps}fps"
        )

    def _init_camera(self):
        """Initialize the camera."""
        if PICAMERA_AVAILABLE:
            try:
                self.camera = Picamera2()

                # Configure for video
                config = self.camera.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    controls={"FrameRate": self.fps}
                )
                self.camera.configure(config)
                self.camera.start()

                self.get_logger().info("Pi Camera initialized successfully")
                self.use_picamera = True
                return
            except Exception as e:
                self.get_logger().warn(f"Failed to init Pi Camera: {e}")

        # Fallback to OpenCV
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)

            if self.camera.isOpened():
                self.get_logger().info("OpenCV camera initialized")
                self.use_picamera = False
            else:
                self.get_logger().error("Failed to open camera")
                self.camera = None
        except Exception as e:
            self.get_logger().error(f"Camera init error: {e}")
            self.camera = None

    def timer_callback(self):
        """Capture and publish camera frame."""
        if self.camera is None:
            return

        try:
            # Capture frame
            if self.use_picamera:
                frame = self.camera.capture_array()
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = self.camera.read()
                if not ret:
                    return

            # Compress to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            _, compressed = cv2.imencode('.jpg', frame, encode_param)

            # Create and publish message
            msg = CompressedImage()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_link'
            msg.format = 'jpeg'
            msg.data = compressed.tobytes()

            self.image_pub.publish(msg)

            # Publish camera info
            self._publish_camera_info(msg.header)

            # Update stats
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed
                self.get_logger().info(f"Streaming at {fps:.1f} FPS")

        except Exception as e:
            self.get_logger().error(f"Frame capture error: {e}")

    def _publish_camera_info(self, header):
        """Publish camera info message."""
        info = CameraInfo()
        info.header = header
        info.height = self.height
        info.width = self.width

        # Pi Camera Wide approximate intrinsics
        # These should be calibrated for accurate results
        fx = self.width * 0.8
        fy = self.width * 0.8
        cx = self.width / 2
        cy = self.height / 2

        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        self.camera_info_pub.publish(info)

    def destroy_node(self):
        """Clean up on shutdown."""
        if self.camera is not None:
            if self.use_picamera:
                self.camera.stop()
            else:
                self.camera.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraStreamerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
