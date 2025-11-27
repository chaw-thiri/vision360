#!/usr/bin/env python3
"""
Test individual wheel movements to diagnose hardware issues
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class WheelTest(Node):
    def __init__(self):
        super().__init__('wheel_test')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Wheel Diagnostic Test Started')

    def test_command(self, linear, angular, duration, description):
        """Send a test command."""
        self.get_logger().info(f'\n{description}')
        self.get_logger().info(f'  Command: Linear={linear:.2f} m/s, Angular={angular:.2f} rad/s')

        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular

        start_time = time.time()
        while (time.time() - start_time) < duration:
            self.publisher.publish(msg)
            time.sleep(0.1)

        # Stop
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.publisher.publish(msg)
        time.sleep(1)

    def run_diagnostic(self):
        """Run wheel diagnostic tests."""
        self.get_logger().info('='*60)
        self.get_logger().info('TurtleBot3 Wheel Diagnostic Test')
        self.get_logger().info('='*60)
        self.get_logger().info('\nThis will test each movement to identify wheel issues.')
        time.sleep(2)

        try:
            # Test 1: Forward (both wheels forward)
            self.test_command(0.1, 0.0, 2.0,
                "TEST 1: Forward - Both wheels should move forward")

            # Test 2: Rotate Left (right wheel forward, left wheel backward)
            self.test_command(0.0, 0.5, 2.0,
                "TEST 2: Rotate Left - Right wheel forward, Left wheel backward")

            # Test 3: Rotate Right (left wheel forward, right wheel backward)
            self.test_command(0.0, -0.5, 2.0,
                "TEST 3: Rotate Right - Left wheel forward, Right wheel backward")

            # Test 4: Backward (both wheels backward)
            self.test_command(-0.1, 0.0, 2.0,
                "TEST 4: Backward - Both wheels should move backward")

            # Test 5: Arc Left (right wheel faster than left)
            self.test_command(0.1, 0.3, 2.0,
                "TEST 5: Arc Left - Right wheel faster, left wheel slower")

            # Test 6: Arc Right (left wheel faster than right)
            self.test_command(0.1, -0.3, 2.0,
                "TEST 6: Arc Right - Left wheel faster, right wheel slower")

            # Final stop
            msg = Twist()
            self.publisher.publish(msg)

            self.get_logger().info('\n' + '='*60)
            self.get_logger().info('Diagnostic Complete!')
            self.get_logger().info('='*60)
            self.get_logger().info('\nREPORT YOUR OBSERVATIONS:')
            self.get_logger().info('  - Which tests showed the left wheel not moving?')
            self.get_logger().info('  - Does the left wheel make any sound?')
            self.get_logger().info('  - Is the wheel physically obstructed?')

        except KeyboardInterrupt:
            self.get_logger().info('Test interrupted')
            msg = Twist()
            self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    try:
        tester = WheelTest()
        tester.run_diagnostic()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
