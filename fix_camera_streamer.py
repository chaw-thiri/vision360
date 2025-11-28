#!/usr/bin/env python3
"""
Fixed Camera Streamer that tries multiple video devices
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

import cv2
import numpy as np


class CameraStreamerNode(Node):
    """ROS 2 node for streaming camera with auto-detection."""

    def __init__(self):
        super().__init__('camera_streamer_node')

        # Parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 15)  # Lower FPS for stability
        self.declare_parameter('jpeg_quality', 75)

        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value

        # QoS profile
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

        if self.camera is not None:
            # Create timer for publishing
            timer_period = 1.0 / self.fps
            self.timer = self.create_timer(timer_period, self.timer_callback)

            # Stats
            self.frame_count = 0
            self.start_time = time.time()

            self.get_logger().info(
                f"Camera streamer ready: {self.width}x{self.height} @ {self.fps}fps"
            )
        else:
            self.get_logger().error("No camera available - exiting")
            rclpy.shutdown()

    def _init_camera(self):
        """Initialize camera by trying multiple video devices."""
        # Try common video devices
        devices_to_try = [0, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 31]

        for device_id in devices_to_try:
            try:
                self.get_logger().info(f"Trying /dev/video{device_id}...")
                cap = cv2.VideoCapture(device_id)

                if cap.isOpened():
                    # Try to read a test frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.get_logger().info(f"✓ Camera found at /dev/video{device_id}")

                        # Configure camera
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        cap.set(cv2.CAP_PROP_FPS, self.fps)

                        # Verify settings
                        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        self.get_logger().info(f"Camera resolution: {actual_width}x{actual_height}")

                        self.camera = cap
                        return
                    else:
                        cap.release()
                else:
                    cap.release()

            except Exception as e:
                self.get_logger().debug(f"Device {device_id} error: {e}")
                continue

        self.get_logger().error("No working camera found!")

    def timer_callback(self):
        """Capture and publish camera frame."""
        if self.camera is None:
            return

        try:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                self.get_logger().warn("Failed to capture frame")
                return

            # Resize if needed
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

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
            if self.frame_count % 50 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed
                self.get_logger().info(f"Streaming at {fps:.1f} FPS ({len(compressed)} bytes)")

        except Exception as e:
            self.get_logger().error(f"Frame capture error: {e}")

    def _publish_camera_info(self, header):
        """Publish camera info message."""
        info = CameraInfo()
        info.header = header
        info.height = self.height
        info.width = self.width

        # Approximate intrinsics
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
