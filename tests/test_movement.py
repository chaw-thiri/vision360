#!/usr/bin/env python3
"""
Simple Movement Test for TurtleBot3
Tests basic movements in sequence
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class MovementTest(Node):
    def __init__(self):
        super().__init__('movement_test')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Movement Test Node Started')

    def move(self, linear, angular, duration, description):
        """Move the robot with given velocities for specified duration."""
        self.get_logger().info(f'Testing: {description}')
        self.get_logger().info(f'  Linear: {linear} m/s, Angular: {angular} rad/s for {duration}s')

        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular

        # Publish at 10Hz for the duration
        start_time = time.time()
        while (time.time() - start_time) < duration:
            self.publisher.publish(msg)
            time.sleep(0.1)  # 10Hz

        # Stop
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.publisher.publish(msg)
        self.get_logger().info('  Stopped')

        # Wait between movements
        time.sleep(1)

    def run_test_sequence(self):
        """Run a sequence of test movements."""
        self.get_logger().info('='*50)
        self.get_logger().info('Starting TurtleBot Movement Test Sequence')
        self.get_logger().info('='*50)
        time.sleep(2)

        try:
            # Test 1: Forward
            self.move(0.1, 0.0, 2.0, "Move Forward")

            # Test 2: Backward
            self.move(-0.1, 0.0, 2.0, "Move Backward")

            # Test 3: Rotate Left
            self.move(0.0, 0.5, 2.0, "Rotate Left")

            # Test 4: Rotate Right
            self.move(0.0, -0.5, 2.0, "Rotate Right")

            # Test 5: Circle (forward + rotate)
            self.move(0.1, 0.3, 3.0, "Move in Circle")

            # Final stop
            msg = Twist()
            self.publisher.publish(msg)

            self.get_logger().info('='*50)
            self.get_logger().info('Test Complete! Robot stopped.')
            self.get_logger().info('='*50)

        except KeyboardInterrupt:
            self.get_logger().info('Test interrupted by user')
            msg = Twist()
            self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    try:
        tester = MovementTest()
        tester.run_test_sequence()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
