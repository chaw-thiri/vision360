"""
Velocity Controller for TurtleBot3.
Handles velocity smoothing and safety limits.
"""

import time
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Velocity:
    """Velocity command."""
    linear: float  # m/s
    angular: float  # rad/s


class VelocityController:
    """
    Velocity controller with smoothing and safety features.
    Provides smooth acceleration/deceleration and enforces limits.
    """

    def __init__(self, config: dict):
        """
        Initialize velocity controller.

        Args:
            config: Configuration dictionary from config.yaml
        """
        tb_config = config.get('turtlebot', {})

        # Velocity limits
        self.max_linear = tb_config.get('max_linear_velocity', 0.22)
        self.max_angular = tb_config.get('max_angular_velocity', 2.84)

        # Acceleration limits (m/s^2 and rad/s^2)
        self.max_linear_accel = 0.5
        self.max_angular_accel = 2.0

        # Deceleration (faster than acceleration for safety)
        self.max_linear_decel = 1.0
        self.max_angular_decel = 4.0

        # Current velocity state
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.last_update_time = time.time()

        # Emergency stop state
        self.emergency_stop_active = False

    def compute(self, target_linear: float, target_angular: float) -> Velocity:
        """
        Compute smoothed velocity command.

        Args:
            target_linear: Target linear velocity (m/s)
            target_angular: Target angular velocity (rad/s)

        Returns:
            Smoothed Velocity command
        """
        # Handle emergency stop
        if self.emergency_stop_active:
            return self._emergency_stop()

        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        if dt <= 0:
            dt = 0.01

        # Clamp targets to limits
        target_linear = max(-self.max_linear, min(self.max_linear, target_linear))
        target_angular = max(-self.max_angular, min(self.max_angular, target_angular))

        # Smooth linear velocity
        self.current_linear = self._smooth_velocity(
            self.current_linear, target_linear, dt,
            self.max_linear_accel, self.max_linear_decel
        )

        # Smooth angular velocity
        self.current_angular = self._smooth_velocity(
            self.current_angular, target_angular, dt,
            self.max_angular_accel, self.max_angular_decel
        )

        return Velocity(
            linear=self.current_linear,
            angular=self.current_angular
        )

    def _smooth_velocity(self, current: float, target: float, dt: float,
                         max_accel: float, max_decel: float) -> float:
        """Apply velocity smoothing with acceleration limits."""
        diff = target - current

        if abs(diff) < 0.001:
            return target

        # Determine if accelerating or decelerating
        if abs(target) > abs(current) or (target * current < 0):
            # Accelerating or changing direction
            max_change = max_accel * dt
        else:
            # Decelerating
            max_change = max_decel * dt

        # Apply change limit
        if abs(diff) > max_change:
            change = max_change if diff > 0 else -max_change
        else:
            change = diff

        return current + change

    def _emergency_stop(self) -> Velocity:
        """Execute emergency stop with maximum deceleration."""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        if dt <= 0:
            dt = 0.01

        # Rapid deceleration
        decel_rate = self.max_linear_decel * 2  # Double decel for emergency
        angular_decel_rate = self.max_angular_decel * 2

        # Decelerate linear
        if abs(self.current_linear) > 0.001:
            change = decel_rate * dt
            if self.current_linear > 0:
                self.current_linear = max(0, self.current_linear - change)
            else:
                self.current_linear = min(0, self.current_linear + change)
        else:
            self.current_linear = 0.0

        # Decelerate angular
        if abs(self.current_angular) > 0.001:
            change = angular_decel_rate * dt
            if self.current_angular > 0:
                self.current_angular = max(0, self.current_angular - change)
            else:
                self.current_angular = min(0, self.current_angular + change)
        else:
            self.current_angular = 0.0

        return Velocity(
            linear=self.current_linear,
            angular=self.current_angular
        )

    def trigger_emergency_stop(self):
        """Trigger emergency stop."""
        self.emergency_stop_active = True

    def release_emergency_stop(self):
        """Release emergency stop."""
        self.emergency_stop_active = False

    def stop(self):
        """Immediate stop (sets velocities to zero)."""
        self.current_linear = 0.0
        self.current_angular = 0.0

    def is_stopped(self) -> bool:
        """Check if robot is stopped."""
        return abs(self.current_linear) < 0.001 and abs(self.current_angular) < 0.001

    def get_current_velocity(self) -> Velocity:
        """Get current velocity state."""
        return Velocity(
            linear=self.current_linear,
            angular=self.current_angular
        )
