#!/usr/bin/env python3
"""
Simple camera viewer - displays live feed from TurtleBot camera
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')

        # QoS profile for best effort
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to compressed image
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            qos_profile
        )

        self.get_logger().info('Camera Viewer Started')
        self.get_logger().info('Press Q to quit')

        self.frame_count = 0

    def image_callback(self, msg):
        """Display incoming camera image."""
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                self.get_logger().warn('Failed to decode image')
                return

            # Add frame info
            self.frame_count += 1
            cv2.putText(frame, f'Frame: {self.frame_count}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, 'Press Q to quit',
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display
            cv2.imshow('TurtleBot Camera - Live Feed', frame)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Quit requested')
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f'Error displaying image: {e}')

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    print("="*60)
    print("TurtleBot Camera Viewer")
    print("="*60)
    print("Waiting for camera feed from /camera/image_raw/compressed")
    print("Press 'Q' to quit")
    print("="*60)

    rclpy.init()

    try:
        viewer = CameraViewer()
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
