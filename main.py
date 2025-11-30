#!/usr/bin/env python3
"""
Autonomous Driving Vision System - Main Entry Point

This application performs:
- Pedestrian detection using YOLOv8
- Lane detection for indoor taped tracks
- Traffic light detection
- Decision making for autonomous navigation

Can run in two modes:
1. ROS 2 mode: Full integration with TurtleBot3
2. Standalone mode: Testing with webcam or video file
"""

import argparse
import sys
import os
import time
import cv2
import numpy as np
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detectors.person_detector import PersonDetector
from src.detectors.lane_detector import LaneDetector
from src.detectors.traffic_sign_light_detector import TrafficSignLightDetector, TrafficLightState
from src.detectors.boundary_platform_detector import BoundaryPlatformDetector
from src.controller.decision_maker import DecisionMaker
from src.controller.velocity_controller import VelocityController


class AutonomousDrivingSystem:
    """Main autonomous driving vision system."""

    def __init__(self, config_path: str = None):
        """Initialize the system."""
        self.config = self._load_config(config_path)

        print("Initializing Autonomous Driving Vision System...")

        # Initialize detectors
        print("  - Loading person detector (YOLOv8)...")
        self.person_detector = PersonDetector(self.config)

        print("  - Initializing lane detector...")
        self.lane_detector = LaneDetector(self.config)

        print("  - Initializing traffic sign and light detector...")
        self.traffic_sign_light_detector = TrafficSignLightDetector(self.config)

        print("  - Initializing boundary platform detector...")
        self.boundary_detector = BoundaryPlatformDetector(self.config)

        # Initialize controllers
        print("  - Initializing decision maker...")
        self.decision_maker = DecisionMaker(self.config)
        self.velocity_controller = VelocityController(self.config)

        # Visualization settings
        viz_config = self.config.get('visualization', {})
        self.show_windows = viz_config.get('show_windows', True)
        self.show_debug = False

        # Manual control state
        self.manual_override_active = False
        self.manual_linear = 0.0
        self.manual_angular = 0.0
        self.manual_speed = 0.15  # Default manual speed

        # FPS tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        print("System initialized successfully!")

    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / 'config' / 'config.yaml'

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded config from: {config_path}")
            return config
        except Exception as e:
            print(f"Warning: Could not load config ({e}), using defaults")
            return {}

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame through all detectors.

        Args:
            frame: BGR image

        Returns:
            Tuple of (output_frame, command)
        """
        if frame is None:
            return None, None

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
            velocity_linear = self.manual_linear
            velocity_angular = self.manual_angular
            # Create dummy command for visualization
            from src.controller.decision_maker import NavigationCommand, RobotState, StopReason
            command = NavigationCommand(
                linear_velocity=velocity_linear,
                angular_velocity=velocity_angular,
                state=RobotState.MOVING if velocity_linear != 0 or velocity_angular != 0 else RobotState.STOPPED,
                stop_reason=StopReason.MANUAL,
                message="MANUAL CONTROL ACTIVE"
            )
        else:
            # Make decision with boundary platform information
            command = self.decision_maker.decide(
                person_danger,
                avoidance,
                lane_info,
                traffic_state,
                boundary_info.all_blocked,
                boundary_info.avoidance_angle
            )

            # Apply velocity smoothing
            velocity = self.velocity_controller.compute(
                command.linear_velocity,
                command.angular_velocity
            )
            velocity_linear = velocity.linear
            velocity_angular = velocity.angular

        # Create visualization
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
        self._draw_info(output, command, velocity_linear, velocity_angular)

        # Update FPS
        self._update_fps()

        return output, command

    def _draw_info(self, frame: np.ndarray, command, linear: float, angular: float):
        """Draw system information on frame."""
        height, width = frame.shape[:2]

        # Background box
        cv2.rectangle(frame, (10, height - 130), (320, height - 10),
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (10, height - 130), (320, height - 10),
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
        if abs(angular) < 0.05:
            nav_direction = "STRAIGHT"
            dir_color = (0, 255, 0)
        elif angular > 0:
            nav_direction = "LEFT"
            dir_color = (255, 0, 255)
        else:
            nav_direction = "RIGHT"
            dir_color = (255, 255, 0)

        # Info text
        y_offset = height - 110
        line_height = 20

        cv2.putText(frame, f"State: {command.state.value.upper()}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_offset += line_height

        cv2.putText(frame, f"Navigate: {nav_direction}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dir_color, 2)
        y_offset += line_height

        cv2.putText(frame, f"Speed: {linear:.3f} m/s",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += line_height

        cv2.putText(frame, f"Turn: {angular:+.3f} rad/s",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # FPS in top right
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}",
                   (width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Controls help
        cv2.putText(frame, "Arrows:Move Space:Resume Q:Stop R:Resume X:Quit",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time

        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()

    def run_webcam(self, camera_index: int = 0):
        """Run system with webcam input."""
        print(f"\nStarting webcam capture (index {camera_index})...")

        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        # Set resolution
        cam_config = self.config.get('camera', {})
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config.get('width', 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config.get('height', 480))

        print("Webcam started. Press 'x' to quit.")
        print("Controls: Q=Stop, R=Resume, X=Quit, D=Toggle Debug")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Process frame
                output, command = self.process_frame(frame)

                if output is not None:
                    cv2.imshow("Autonomous Driving Vision", output)

                    # Show debug view if enabled
                    if self.show_debug:
                        debug = self.lane_detector.get_debug_view(frame)
                        cv2.imshow("Debug - Lane Detection", debug)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                # Arrow keys for manual control
                if key == 82 or key == ord('w'):  # Up arrow or W
                    self.manual_override_active = True
                    self.manual_linear = self.manual_speed
                    self.manual_angular = 0.0
                    print("Manual: FORWARD")
                elif key == 84 or key == ord('s'):  # Down arrow or S
                    self.manual_override_active = True
                    self.manual_linear = -self.manual_speed
                    self.manual_angular = 0.0
                    print("Manual: BACKWARD")
                elif key == 81 or key == ord('a'):  # Left arrow or A
                    self.manual_override_active = True
                    self.manual_linear = 0.0
                    self.manual_angular = 0.5
                    print("Manual: TURN LEFT")
                elif key == 83 or key == ord('d'):  # Right arrow or D
                    self.manual_override_active = True
                    self.manual_linear = 0.0
                    self.manual_angular = -0.5
                    print("Manual: TURN RIGHT")
                elif key == ord(' '):  # Spacebar - stop manual and resume auto
                    self.manual_override_active = False
                    self.manual_linear = 0.0
                    self.manual_angular = 0.0
                    self.decision_maker.resume()
                    print("Resumed AUTONOMOUS control")
                elif key == ord('q'):
                    # Q key - emergency stop
                    self.manual_override_active = False
                    self.manual_linear = 0.0
                    self.manual_angular = 0.0
                    self.decision_maker.emergency_stop()
                    print("Q pressed - EMERGENCY STOP")
                elif key == ord('r'):
                    # R key - resume autonomous
                    self.manual_override_active = False
                    self.manual_linear = 0.0
                    self.manual_angular = 0.0
                    self.decision_maker.resume()
                    print("R pressed - RESUMING autonomous control")
                elif key == ord('x'):
                    print("X pressed - QUITTING")
                    break
                elif key == ord('+') or key == ord('='):
                    self.manual_speed = min(0.22, self.manual_speed + 0.02)
                    print(f"Manual speed: {self.manual_speed:.2f} m/s")
                elif key == ord('-') or key == ord('_'):
                    self.manual_speed = max(0.05, self.manual_speed - 0.02)
                    print(f"Manual speed: {self.manual_speed:.2f} m/s")
                elif key == ord('t'):  # Toggle debug (changed from 'd' to 't')
                    self.show_debug = not self.show_debug
                    if not self.show_debug:
                        cv2.destroyWindow("Debug - Lane Detection")
                    print(f"Debug view: {'ON' if self.show_debug else 'OFF'}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def run_video(self, video_path: str):
        """Run system with video file input."""
        print(f"\nProcessing video: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int(1000 / fps) if fps > 0 else 33

        print(f"Video FPS: {fps:.1f}")
        print("Press 'q' to quit, SPACE to pause")

        paused = False

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video")
                        break

                    output, command = self.process_frame(frame)

                    if output is not None:
                        cv2.imshow("Autonomous Driving Vision", output)

                key = cv2.waitKey(frame_delay if not paused else 0) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Playing'}")
                elif key == ord('s'):
                    self.decision_maker.emergency_stop()
                elif key == ord('r'):
                    self.decision_maker.resume()

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def run_image(self, image_path: str):
        """Run system on single image."""
        print(f"\nProcessing image: {image_path}")

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image: {image_path}")
            return

        output, command = self.process_frame(frame)

        if output is not None:
            cv2.imshow("Autonomous Driving Vision", output)
            print("\nPress any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Print command info
        print(f"\nDetection Results:")
        print(f"  State: {command.state.value}")
        print(f"  Linear Velocity: {command.linear_velocity:.3f} m/s")
        print(f"  Angular Velocity: {command.angular_velocity:.3f} rad/s")
        print(f"  Message: {command.message}")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Driving Vision System for TurtleBot3"
    )

    parser.add_argument(
        '--mode', type=str, default='webcam',
        choices=['webcam', 'video', 'image', 'ros'],
        help='Operation mode (default: webcam)'
    )

    parser.add_argument(
        '--input', type=str, default=None,
        help='Input file path (for video/image mode)'
    )

    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera index for webcam mode (default: 0)'
    )

    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Create system
    system = AutonomousDrivingSystem(args.config)

    # Run based on mode
    if args.mode == 'webcam':
        system.run_webcam(args.camera)

    elif args.mode == 'video':
        if args.input is None:
            print("Error: --input required for video mode")
            sys.exit(1)
        system.run_video(args.input)

    elif args.mode == 'image':
        if args.input is None:
            print("Error: --input required for image mode")
            sys.exit(1)
        system.run_image(args.input)

    elif args.mode == 'ros':
        try:
            from src.ros_nodes.vision_node import main as ros_main
            ros_main()
        except ImportError as e:
            print(f"Error: ROS 2 not available ({e})")
            print("Please install ROS 2 Humble or use standalone mode")
            sys.exit(1)


if __name__ == '__main__':
    main()
