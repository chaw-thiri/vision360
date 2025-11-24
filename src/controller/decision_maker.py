"""
Decision Making Module for Autonomous Navigation.
Combines inputs from all detectors to determine robot actions.
"""

import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

from ..detectors.person_detector import DetectedPerson
from ..detectors.lane_detector import LaneInfo
from ..detectors.traffic_light_detector import TrafficLightState, DetectedTrafficLight


class RobotState(Enum):
    """Robot operational states."""
    STOPPED = "stopped"
    MOVING = "moving"
    SLOWING = "slowing"
    EMERGENCY_STOP = "emergency_stop"


class StopReason(Enum):
    """Reasons for stopping."""
    NONE = "none"
    PEDESTRIAN = "pedestrian"
    RED_LIGHT = "red_light"
    YELLOW_LIGHT = "yellow_light"
    NO_LANE = "no_lane"
    MANUAL = "manual"


@dataclass
class NavigationCommand:
    """Navigation command output."""
    linear_velocity: float  # m/s
    angular_velocity: float  # rad/s
    state: RobotState
    stop_reason: StopReason
    message: str


class PIDController:
    """Simple PID controller for lane following."""

    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, error: float) -> float:
        """Compute PID output."""
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0:
            dt = 0.01

        # Proportional
        p = self.kp * error

        # Integral
        self.integral += error * dt
        self.integral = max(-1.0, min(1.0, self.integral))  # Anti-windup
        i = self.ki * self.integral

        # Derivative
        derivative = (error - self.prev_error) / dt
        d = self.kd * derivative

        # Update state
        self.prev_error = error
        self.last_time = current_time

        return p + i + d

    def reset(self):
        """Reset controller state."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


class DecisionMaker:
    """
    Decision maker for autonomous navigation.
    Combines person detection, lane detection, and traffic light detection
    to make driving decisions.
    """

    def __init__(self, config: dict):
        """
        Initialize decision maker.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config

        # TurtleBot settings
        tb_config = config.get('turtlebot', {})
        self.max_linear = tb_config.get('max_linear_velocity', 0.22)
        self.max_angular = tb_config.get('max_angular_velocity', 2.84)
        self.normal_speed = tb_config.get('normal_speed', 0.15)
        self.slow_speed = tb_config.get('slow_speed', 0.08)

        # Decision settings
        dec_config = config.get('decision', {})
        self.max_steering = dec_config.get('max_steering_angle', 0.5)

        # PID controller for lane following
        pid_config = dec_config.get('lane_pid', {})
        self.lane_pid = PIDController(
            kp=pid_config.get('kp', 0.005),
            ki=pid_config.get('ki', 0.0001),
            kd=pid_config.get('kd', 0.001)
        )

        # State tracking
        self.current_state = RobotState.STOPPED
        self.stop_reason = StopReason.NONE
        self.manual_stop = False

    def decide(self,
               person_danger: str,
               person_avoidance: Optional[str],
               lane_info: LaneInfo,
               traffic_state: TrafficLightState) -> NavigationCommand:
        """
        Make navigation decision based on all inputs.

        Priority order:
        1. Pedestrian safety (highest)
        2. Traffic lights
        3. Lane following (lowest)

        Args:
            person_danger: Danger level from person detector
            person_avoidance: Suggested avoidance direction
            lane_info: Lane detection results
            traffic_state: Current traffic light state

        Returns:
            NavigationCommand with velocities and state
        """
        # Manual stop override
        if self.manual_stop:
            return NavigationCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
                state=RobotState.STOPPED,
                stop_reason=StopReason.MANUAL,
                message="Manual stop engaged"
            )

        # Priority 1: Pedestrian safety
        if person_danger == 'stop':
            self.lane_pid.reset()
            return NavigationCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
                state=RobotState.EMERGENCY_STOP,
                stop_reason=StopReason.PEDESTRIAN,
                message="EMERGENCY STOP: Pedestrian too close!"
            )

        if person_danger == 'danger':
            # Slow down and try to avoid
            angular = 0.0
            if person_avoidance == 'left':
                angular = self.max_angular * 0.3
            elif person_avoidance == 'right':
                angular = -self.max_angular * 0.3

            return NavigationCommand(
                linear_velocity=self.slow_speed * 0.5,
                angular_velocity=angular,
                state=RobotState.SLOWING,
                stop_reason=StopReason.PEDESTRIAN,
                message=f"Caution: Pedestrian detected, avoiding {person_avoidance or 'ahead'}"
            )

        # Priority 2: Traffic lights
        if traffic_state == TrafficLightState.RED:
            self.lane_pid.reset()
            return NavigationCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
                state=RobotState.STOPPED,
                stop_reason=StopReason.RED_LIGHT,
                message="Stopped at red light"
            )

        if traffic_state == TrafficLightState.YELLOW:
            # Slow down for yellow light
            angular = self._compute_lane_steering(lane_info)
            return NavigationCommand(
                linear_velocity=self.slow_speed,
                angular_velocity=angular,
                state=RobotState.SLOWING,
                stop_reason=StopReason.YELLOW_LIGHT,
                message="Slowing for yellow light"
            )

        # Priority 3: Lane following
        if not lane_info.lane_detected:
            # No lane detected - slow down and go straight
            return NavigationCommand(
                linear_velocity=self.slow_speed * 0.5,
                angular_velocity=0.0,
                state=RobotState.SLOWING,
                stop_reason=StopReason.NO_LANE,
                message="No lane detected - proceeding slowly"
            )

        # Normal lane following
        angular = self._compute_lane_steering(lane_info)

        # Adjust speed based on steering (slow down in curves)
        speed_factor = 1.0 - abs(angular) / self.max_angular * 0.5
        linear = self.normal_speed * speed_factor

        # Slight speed reduction if pedestrian in caution zone
        if person_danger == 'caution':
            linear *= 0.8

        return NavigationCommand(
            linear_velocity=linear,
            angular_velocity=angular,
            state=RobotState.MOVING,
            stop_reason=StopReason.NONE,
            message=f"Following lane (offset: {lane_info.center_offset:+.2f})"
        )

    def _compute_lane_steering(self, lane_info: LaneInfo) -> float:
        """Compute steering angle using PID controller."""
        if not lane_info.lane_detected:
            return 0.0

        # Use center offset as error
        error = lane_info.center_offset

        # Compute PID output
        steering = self.lane_pid.compute(error)

        # Combine with lane detector's suggested steering
        steering = steering * 0.7 + lane_info.steering_angle * 0.3

        # Clamp to limits
        steering = max(-self.max_steering, min(self.max_steering, steering))

        # Convert to angular velocity
        angular = steering * self.max_angular

        return angular

    def emergency_stop(self):
        """Trigger emergency stop."""
        self.manual_stop = True
        self.current_state = RobotState.EMERGENCY_STOP

    def resume(self):
        """Resume from manual stop."""
        self.manual_stop = False
        self.lane_pid.reset()

    def get_status_text(self, command: NavigationCommand) -> List[str]:
        """Get status text for visualization."""
        return [
            f"State: {command.state.value}",
            f"Linear: {command.linear_velocity:.3f} m/s",
            f"Angular: {command.angular_velocity:.3f} rad/s",
            f"Stop Reason: {command.stop_reason.value}",
            f"Message: {command.message}"
        ]
