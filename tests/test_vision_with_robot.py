#!/usr/bin/env python3
"""
Vision360 Test Mode: Video Input + Real Robot Control
Tests the complete vision system using video while controlling the TurtleBot
"""

import sys
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pathlib import Path
import yaml
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.detectors.person_detector import PersonDetector
from src.detectors.lane_detector import LaneDetector
from src.detectors.traffic_light_detector import TrafficLightDetector
from src.controller.decision_maker import DecisionMaker
from src.controller.velocity_controller import VelocityController


class VideoRobotTestNode(Node):
    """Test vision system with video input while controlling real robot."""

    def __init__(self, video_path, config_path=None):
        super().__init__('video_robot_test')

        # Load config
        if config_path is None:
            config_path = Path(__file__).parent / 'config' / 'config.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.get_logger().info("="*60)
        self.get_logger().info("Vision360 Test Mode - Video + Real Robot")
        self.get_logger().info("="*60)
        self.get_logger().info("⚠️  WARNING: Robot will move based on video!")
        self.get_logger().info("   Press 'S' for emergency stop")
        self.get_logger().info("   Press 'Q' to quit")
        self.get_logger().info("="*60)

        # Initialize detectors
        self.get_logger().info("Loading vision detectors...")
        self.person_detector = PersonDetector(self.config)
        self.lane_detector = LaneDetector(self.config)
        self.traffic_detector = TrafficLightDetector(self.config)

        # Initialize controllers
        self.decision_maker = DecisionMaker(self.config)
        self.velocity_controller = VelocityController(self.config)

        # Publisher for robot control
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video: {video_path}")
            rclpy.shutdown()
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_delay = 1.0 / self.fps if self.fps > 0 else 0.033

        self.get_logger().info(f"✓ Video loaded: {video_path}")
        self.get_logger().info(f"✓ Video FPS: {self.fps:.1f}")
        self.get_logger().info(f"✓ Publishing to /cmd_vel")

        # Stats
        self.frame_count = 0
        self.start_time = time.time()

        # Create timer for processing
        self.timer = self.create_timer(self.frame_delay, self.process_frame)

        self.get_logger().info("System ready! Robot will start moving...")
        time.sleep(2)

    def process_frame(self):
        """Process one frame and control robot."""
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().info("Video ended - restarting...")
            self.publish_velocity(0.0, 0.0)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            return

        height, width = frame.shape[:2]

        # Run all vision detectors
        person_detections = self.person_detector.detect(frame)
        person_danger, closest_person, avoidance = self.person_detector.get_danger_level(
            person_detections, width
        )

        lane_info = self.lane_detector.detect(frame)
        traffic_state, traffic_detections = self.traffic_detector.detect(frame)

        # Make driving decision
        command = self.decision_maker.decide(
            person_danger,
            avoidance,
            lane_info,
            traffic_state
        )

        # Apply velocity smoothing
        velocity = self.velocity_controller.compute(
            command.linear_velocity,
            command.angular_velocity
        )

        # Publish to REAL robot
        self.publish_velocity(velocity.linear, velocity.angular)

        # Log important events
        # Check if there's any danger (handles both int and string types)
        has_danger = (person_danger != 0 and person_danger != "none" and person_danger is not None)
        if has_danger or command.state.value != 'moving':
            self.get_logger().info(
                f"🚨 {command.state.value.upper()} | "
                f"Pedestrian: {person_danger} | "
                f"Traffic: {traffic_state.value} | "
                f"Cmd: {velocity.linear:.2f}, {velocity.angular:.2f}"
            )

        # Visualization
        output = self.visualize(frame, person_detections, person_danger,
                               lane_info, traffic_state, traffic_detections, command, velocity)

        cv2.imshow("Vision360 Test - Controlling Real Robot!", output)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.get_logger().info("Quit requested - stopping robot")
            self.publish_velocity(0.0, 0.0)
            rclpy.shutdown()
        elif key == ord('s'):
            self.decision_maker.emergency_stop()
            self.get_logger().warn("🛑 EMERGENCY STOP!")
        elif key == ord('r'):
            self.decision_maker.resume()
            self.get_logger().info("▶️  Resumed")

        self.frame_count += 1

    def publish_velocity(self, linear, angular):
        """Publish velocity command to robot."""
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self.cmd_vel_pub.publish(twist)

    def visualize(self, frame, person_detections, person_danger,
                  lane_info, traffic_state, traffic_detections, command, velocity):
        """Create visualization with overlays."""
        output = frame.copy()

        # Draw all detections
        output = self.lane_detector.draw_lanes(output, lane_info)
        output = self.person_detector.draw_detections(output, person_detections, person_danger)
        output = self.traffic_detector.draw_detections(output, traffic_state, traffic_detections)

        height = output.shape[0]

        # Warning banner
        cv2.rectangle(output, (0, 0), (output.shape[1], 40), (0, 0, 255), -1)
        cv2.putText(output, "TEST MODE: Video Input + REAL Robot Control",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Command info box
        cv2.rectangle(output, (10, height - 120), (350, height - 10), (0, 0, 0), -1)
        cv2.rectangle(output, (10, height - 120), (350, height - 10), (255, 255, 255), 2)

        # State color coding
        state_colors = {
            'stopped': (0, 0, 255),
            'moving': (0, 255, 0),
            'slowing': (0, 255, 255),
            'emergency_stop': (0, 0, 255)
        }
        color = state_colors.get(command.state.value, (255, 255, 255))

        y = height - 95
        cv2.putText(output, f"STATE: {command.state.value.upper()}",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y += 22
        cv2.putText(output, f"Robot Linear: {velocity.linear:.3f} m/s",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y += 20
        cv2.putText(output, f"Robot Angular: {velocity.angular:.3f} rad/s",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y += 20
        cv2.putText(output, f"Reason: {command.stop_reason.value}",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # FPS
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            fps = self.frame_count / elapsed
            cv2.putText(output, f"FPS: {fps:.1f}",
                       (output.shape[1] - 100, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Controls
        cv2.putText(output, "Q:Quit | S:Stop | R:Resume",
                   (10, height - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        return output

    def destroy_node(self):
        """Cleanup - STOP ROBOT!"""
        self.get_logger().info("Shutting down - stopping robot")
        self.publish_velocity(0.0, 0.0)
        time.sleep(0.5)  # Ensure stop command is sent
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='london_bus (1).mp4',
                       help='Video file path')
    parser.add_argument('--config', type=str, default=None,
                       help='Config file path')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("⚠️  SAFETY WARNING")
    print("="*60)
    print("The TurtleBot will move based on vision detection from video!")
    print("Make sure there is clear space around the robot.")
    print("Press Ctrl+C or 'S' key to stop anytime.")
    print("="*60)
    input("Press ENTER to start... ")

    rclpy.init()

    try:
        node = VideoRobotTestNode(args.video, args.config)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
