#!/usr/bin/env python3
"""
Simple Manual Control Script for TurtleBot3
Tests basic movement without camera/vision system
"""

import sys
import tty
import termios
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class ManualControl(Node):
    def __init__(self):
        super().__init__('manual_control')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Movement parameters
        self.linear_speed = 0.15  # m/s
        self.angular_speed = 0.5  # rad/s

        self.get_logger().info('Manual Control Initialized')
        self.get_logger().info('=====================================')
        self.get_logger().info('Control Your TurtleBot:')
        self.get_logger().info('  W/↑ : Move Forward')
        self.get_logger().info('  S/↓ : Move Backward')
        self.get_logger().info('  A/← : Turn Left')
        self.get_logger().info('  D/→ : Turn Right')
        self.get_logger().info('  X   : Stop')
        self.get_logger().info('  Q   : Quit')
        self.get_logger().info('  +   : Increase Speed')
        self.get_logger().info('  -   : Decrease Speed')
        self.get_logger().info('=====================================')

    def get_key(self):
        """Get keyboard input."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def publish_velocity(self, linear, angular):
        """Publish velocity command."""
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.publisher.publish(msg)

        # Log the command
        if linear != 0 or angular != 0:
            self.get_logger().info(
                f'Moving - Linear: {linear:.2f} m/s, Angular: {angular:.2f} rad/s'
            )
        else:
            self.get_logger().info('Stopped')

    def stop(self):
        """Send stop command."""
        self.publish_velocity(0.0, 0.0)

    def run(self):
        """Main control loop."""
        try:
            while True:
                key = self.get_key()

                # Convert key to lowercase for easier handling
                key_lower = key.lower()

                # Movement commands
                if key_lower == 'w':
                    self.publish_velocity(self.linear_speed, 0.0)

                elif key_lower == 's':
                    self.publish_velocity(-self.linear_speed, 0.0)

                elif key_lower == 'a':
                    self.publish_velocity(0.0, self.angular_speed)

                elif key_lower == 'd':
                    self.publish_velocity(0.0, -self.angular_speed)

                elif key_lower == 'x':
                    self.stop()

                # Speed adjustment
                elif key == '+' or key == '=':
                    self.linear_speed = min(0.22, self.linear_speed + 0.01)
                    self.angular_speed = min(2.0, self.angular_speed + 0.1)
                    self.get_logger().info(
                        f'Speed increased - Linear: {self.linear_speed:.2f}, Angular: {self.angular_speed:.2f}'
                    )

                elif key == '-' or key == '_':
                    self.linear_speed = max(0.05, self.linear_speed - 0.01)
                    self.angular_speed = max(0.1, self.angular_speed - 0.1)
                    self.get_logger().info(
                        f'Speed decreased - Linear: {self.linear_speed:.2f}, Angular: {self.angular_speed:.2f}'
                    )

                # Quit
                elif key_lower == 'q':
                    self.get_logger().info('Quitting...')
                    self.stop()
                    break

                # Handle arrow keys (special escape sequences)
                elif key == '\x1b':
                    next1 = sys.stdin.read(1)
                    next2 = sys.stdin.read(1)
                    if next1 == '[':
                        if next2 == 'A':  # Up arrow
                            self.publish_velocity(self.linear_speed, 0.0)
                        elif next2 == 'B':  # Down arrow
                            self.publish_velocity(-self.linear_speed, 0.0)
                        elif next2 == 'C':  # Right arrow
                            self.publish_velocity(0.0, -self.angular_speed)
                        elif next2 == 'D':  # Left arrow
                            self.publish_velocity(0.0, self.angular_speed)

        except KeyboardInterrupt:
            self.get_logger().info('Interrupted by user')
        finally:
            self.stop()


def main(args=None):
    print("\n" + "="*50)
    print("TurtleBot3 Manual Control Test")
    print("="*50)

    rclpy.init(args=args)

    try:
        controller = ManualControl()
        controller.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()
        print("\nShutdown complete. Robot stopped.")


if __name__ == '__main__':
    main()
