#!/usr/bin/env python3
"""
ROS 2 Vision Node for Autonomous Driving.
Main node that processes camera images and publishes velocity commands.
"""

import os
import sys
import cv2
import numpy as np
import yaml
from pathlib import Path

# ROS 2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import CompressedImage, Image
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS 2 not available. Install ros-humble-desktop")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.detectors.person_detector import PersonDetector
from src.detectors.lane_detector import LaneDetector
from src.detectors.traffic_sign_light_detector import TrafficSignLightDetector, TrafficLightState
from src.detectors.boundary_platform_detector import BoundaryPlatformDetector
from src.controller.decision_maker import DecisionMaker
from src.controller.velocity_controller import VelocityController


class VisionNode(Node):
    """ROS 2 node for autonomous driving vision system."""

    def __init__(self, config_path: str = None):
        super().__init__('autonomous_vision_node')

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Initialize detectors
        self.get_logger().info("Initializing detectors...")
        self.person_detector = PersonDetector(self.config)
        self.lane_detector = LaneDetector(self.config)
        self.traffic_sign_light_detector = TrafficSignLightDetector(self.config, mode='ros')
        self.boundary_detector = BoundaryPlatformDetector(self.config)

        # Initialize controllers
        self.decision_maker = DecisionMaker(self.config)
        self.velocity_controller = VelocityController(self.config)

        # Get topic names from config
        camera_config = self.config.get('camera', {})
        tb_config = self.config.get('turtlebot', {})

        image_topic = camera_config.get('image_topic', '/camera/image_raw/compressed')
        cmd_vel_topic = tb_config.get('cmd_vel_topic', '/cmd_vel')

        # QoS profile for image subscription
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            CompressedImage,
            image_topic,
            self.image_callback,
            qos_profile
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            cmd_vel_topic,
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/autonomous_driving/status',
            10
        )

        # Publisher for processed image visualization
        self.debug_image_pub = self.create_publisher(
            CompressedImage,
            '/autonomous_driving/debug_image/compressed',
            10
        )

        # Visualization settings
        viz_config = self.config.get('visualization', {})
        self.show_windows = viz_config.get('show_windows', True)

        # Manual control state
        self.manual_override_active = False
        self.manual_linear = 0.0
        self.manual_angular = 0.0
        self.manual_speed = 0.15  # Default manual speed

        # Processing rate limiter
        self.last_process_time = 0
        self.min_process_interval = 1.0 / 30  # Max 30 FPS

        # Frame counter for FPS calculation
        self.frame_count = 0
        self.fps_start_time = self.get_clock().now()
        self.current_fps = 0.0

        self.get_logger().info(f"Vision node initialized")
        self.get_logger().info(f"Subscribing to: {image_topic}")
        self.get_logger().info(f"Publishing to: {cmd_vel_topic}")

    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.get_logger().info(f"Loaded config from: {config_path}")
            return config
        except Exception as e:
            self.get_logger().error(f"Error loading config: {e}")
            return {}

    def image_callback(self, msg: CompressedImage):
        """Process incoming camera image."""
        # Rate limiting
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_process_time < self.min_process_interval:
            return
        self.last_process_time = current_time

        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                self.get_logger().warn("Failed to decode image")
                return

            # Process frame
            self._process_frame(frame)

            # Update FPS
            self._update_fps()

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def _process_frame(self, frame: np.ndarray):
        """Process a single frame through all detectors."""
        height, width = frame.shape[:2]

        # Run detectors
        person_detections = self.person_detector.detect(frame)
        person_danger, closest_person, avoidance = self.person_detector.get_danger_level(
            person_detections, width
        )

        lane_info = self.lane_detector.detect(frame)

        # Detect traffic lights and signs
        traffic_state, traffic_detections, sign_detections = self.traffic_sign_light_detector.detect(frame)

        # Detect boundary platforms
        boundary_info = self.boundary_detector.detect(frame)

        # Check if manual override is active
        if self.manual_override_active:
            # Use manual control commands directly
            self._publish_velocity(self.manual_linear, self.manual_angular)
            # Create dummy command for visualization
            from src.controller.decision_maker import NavigationCommand, RobotState, StopReason
            command = NavigationCommand(
                linear_velocity=self.manual_linear,
                angular_velocity=self.manual_angular,
                state=RobotState.MOVING if self.manual_linear != 0 or self.manual_angular != 0 else RobotState.STOPPED,
                stop_reason=StopReason.MANUAL,
                message="MANUAL CONTROL ACTIVE"
            )
        else:
            # Make decision with boundary platform information and traffic signs
            command = self.decision_maker.decide(
                person_danger,
                avoidance,
                lane_info,
                traffic_state,
                boundary_info.all_blocked,
                boundary_info.avoidance_angle,
                sign_detections
            )

            # Apply velocity smoothing
            velocity = self.velocity_controller.compute(
                command.linear_velocity,
                command.angular_velocity
            )

            # Publish velocity command
            self._publish_velocity(velocity.linear, velocity.angular)

        # Publish status
        self._publish_status(command, person_danger, traffic_state, lane_info)

        # Visualization
        if self.show_windows:
            self._visualize(frame, person_detections, person_danger,
                          lane_info, traffic_state, traffic_detections, sign_detections,
                          command, boundary_info)

    def _publish_velocity(self, linear: float, angular: float):
        """Publish velocity command to TurtleBot."""
        twist = Twist()
        twist.linear.x = float(linear)
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = float(angular)

        self.cmd_vel_pub.publish(twist)

    def _publish_status(self, command, person_danger, traffic_state, lane_info):
        """Publish status message."""
        status_msg = String()
        status_msg.data = (
            f"state:{command.state.value}|"
            f"linear:{command.linear_velocity:.3f}|"
            f"angular:{command.angular_velocity:.3f}|"
            f"pedestrian:{person_danger}|"
            f"traffic:{traffic_state.value}|"
            f"lane:{lane_info.lane_detected}|"
            f"fps:{self.current_fps:.1f}"
        )
        self.status_pub.publish(status_msg)

    def _visualize(self, frame, person_detections, person_danger,
                   lane_info, traffic_state, traffic_detections, sign_detections, command, boundary_info):
        """Create visualization window."""
        output = frame.copy()

        # Draw boundary platform detection first (as background)
        output = self.boundary_detector.draw_detection(output, boundary_info)

        # Draw lane detection
        output = self.lane_detector.draw_lanes(output, lane_info)

        # Draw person detection
        output = self.person_detector.draw_detections(
            output, person_detections, person_danger
        )

        # Draw traffic lights and road signs
        output = self.traffic_sign_light_detector.draw_detections(
            output, traffic_state, traffic_detections, sign_detections
        )

        # Draw command info
        self._draw_command_info(output, command)

        # Draw FPS
        cv2.putText(output, f"FPS: {self.current_fps:.1f}",
                   (output.shape[1] - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add keyboard controls text
        cv2.putText(output, "Arrows:Move Space:Resume Q:Stop R:Resume X:Quit",
                   (10, output.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, (255, 255, 255), 1)

        # Publish the processed image for remote viewing
        try:
            _, encoded_img = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 80])
            debug_msg = CompressedImage()
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            debug_msg.format = "jpeg"
            debug_msg.data = encoded_img.tobytes()
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing debug image: {e}")

        cv2.imshow("Autonomous Driving Vision", output)
        key = cv2.waitKey(1) & 0xFF

        # Handle key presses
        # Arrow keys for manual control
        if key == 82 or key == ord('w'):  # Up arrow or W
            self.manual_override_active = True
            self.manual_linear = self.manual_speed
            self.manual_angular = 0.0
            self.get_logger().info("Manual: FORWARD")
        elif key == 84 or key == ord('s'):  # Down arrow or S
            self.manual_override_active = True
            self.manual_linear = -self.manual_speed
            self.manual_angular = 0.0
            self.get_logger().info("Manual: BACKWARD")
        elif key == 81 or key == ord('a'):  # Left arrow or A
            self.manual_override_active = True
            self.manual_linear = 0.0
            self.manual_angular = 0.5
            self.get_logger().info("Manual: TURN LEFT")
        elif key == 83 or key == ord('d'):  # Right arrow or D
            self.manual_override_active = True
            self.manual_linear = 0.0
            self.manual_angular = -0.5
            self.get_logger().info("Manual: TURN RIGHT")
        elif key == ord(' '):  # Spacebar - stop manual and resume auto
            self.manual_override_active = False
            self.manual_linear = 0.0
            self.manual_angular = 0.0
            self.decision_maker.resume()
            self.get_logger().info("Resumed AUTONOMOUS control")
        elif key == ord('q'):
            # Q key - emergency stop
            self.manual_override_active = False
            self.manual_linear = 0.0
            self.manual_angular = 0.0
            self.decision_maker.emergency_stop()
            self.get_logger().info("Q pressed - EMERGENCY STOP")
        elif key == ord('r'):
            # R key - resume autonomous
            self.manual_override_active = False
            self.manual_linear = 0.0
            self.manual_angular = 0.0
            self.decision_maker.resume()
            self.get_logger().info("R pressed - RESUMING autonomous control")
        elif key == ord('x'):
            self.get_logger().info("X pressed - QUITTING")
            rclpy.shutdown()
        elif key == ord('+') or key == ord('='):
            self.manual_speed = min(0.22, self.manual_speed + 0.02)
            self.get_logger().info(f"Manual speed: {self.manual_speed:.2f} m/s")
        elif key == ord('-') or key == ord('_'):
            self.manual_speed = max(0.05, self.manual_speed - 0.02)
            self.get_logger().info(f"Manual speed: {self.manual_speed:.2f} m/s")

    def _draw_command_info(self, frame: np.ndarray, command):
        """Draw navigation command information."""
        height = frame.shape[0]

        # Background box
        cv2.rectangle(frame, (10, height - 120), (300, height - 10),
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (10, height - 120), (300, height - 10),
                     (255, 255, 255), 1)

        # State color
        state_colors = {
            'stopped': (0, 0, 255),
            'moving': (0, 255, 0),
            'slowing': (0, 255, 255),
            'emergency_stop': (0, 0, 255)
        }
        color = state_colors.get(command.state.value, (255, 255, 255))

        # Determine navigation direction from angular velocity
        if abs(command.angular_velocity) < 0.05:
            nav_direction = "STRAIGHT"
            dir_color = (0, 255, 0)
        elif command.angular_velocity > 0:
            nav_direction = "LEFT"
            dir_color = (255, 0, 255)
        else:
            nav_direction = "RIGHT"
            dir_color = (255, 255, 0)

        # Draw info
        cv2.putText(frame, f"State: {command.state.value.upper()}",
                   (20, height - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"Navigate: {nav_direction}",
                   (20, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dir_color, 2)
        cv2.putText(frame, f"Speed: {command.linear_velocity:.3f} m/s",
                   (20, height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Turn: {command.angular_velocity:+.3f} rad/s",
                   (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    def _update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1
        elapsed = (self.get_clock().now() - self.fps_start_time).nanoseconds / 1e9

        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = self.get_clock().now()

    def destroy_node(self):
        """Clean up on shutdown."""
        # Stop robot
        self._publish_velocity(0.0, 0.0)
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    if not ROS_AVAILABLE:
        print("Error: ROS 2 is not available. Please install ros-humble-desktop")
        return

    rclpy.init(args=args)

    # Get config path from environment or use default
    config_path = os.environ.get('VISION_CONFIG_PATH', None)

    node = VisionNode(config_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
