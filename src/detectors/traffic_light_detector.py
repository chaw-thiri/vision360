"""
Traffic Light Detection Module.
Detects red, yellow, and green traffic lights for autonomous navigation.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class TrafficLightState(Enum):
    """Traffic light states."""
    NONE = "none"
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


@dataclass
class DetectedTrafficLight:
    """Represents a detected traffic light."""
    state: TrafficLightState
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]
    confidence: float
    area: int


class TrafficLightDetector:
    """Color-based traffic light detector."""

    def __init__(self, config: dict):
        """
        Initialize traffic light detector.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config.get('traffic_light', {})

        # ROI settings
        roi = self.config.get('roi', {})
        self.roi_top = roi.get('top_ratio', 0.0)
        self.roi_bottom = roi.get('bottom_ratio', 0.5)
        self.roi_left = roi.get('left_ratio', 0.2)
        self.roi_right = roi.get('right_ratio', 0.8)

        # Detection parameters
        self.min_area = self.config.get('min_area', 100)
        self.min_circularity = self.config.get('min_circularity', 0.5)

        # Setup color ranges
        self._setup_color_ranges()

        # State tracking for stability
        self.state_history = []
        self.history_size = 5

    def _setup_color_ranges(self):
        """Setup HSV color ranges for traffic light colors."""
        # Red (wraps around in HSV, need two ranges)
        red_cfg = self.config.get('red', {})
        self.red_low_1 = np.array([
            red_cfg.get('h_low_1', 0),
            red_cfg.get('s_low', 100),
            red_cfg.get('v_low', 100)
        ])
        self.red_high_1 = np.array([
            red_cfg.get('h_high_1', 10),
            red_cfg.get('s_high', 255),
            red_cfg.get('v_high', 255)
        ])
        self.red_low_2 = np.array([
            red_cfg.get('h_low_2', 160),
            red_cfg.get('s_low', 100),
            red_cfg.get('v_low', 100)
        ])
        self.red_high_2 = np.array([
            red_cfg.get('h_high_2', 180),
            red_cfg.get('s_high', 255),
            red_cfg.get('v_high', 255)
        ])

        # Yellow
        yellow_cfg = self.config.get('yellow', {})
        self.yellow_low = np.array([
            yellow_cfg.get('h_low', 20),
            yellow_cfg.get('s_low', 100),
            yellow_cfg.get('v_low', 100)
        ])
        self.yellow_high = np.array([
            yellow_cfg.get('h_high', 35),
            yellow_cfg.get('s_high', 255),
            yellow_cfg.get('v_high', 255)
        ])

        # Green
        green_cfg = self.config.get('green', {})
        self.green_low = np.array([
            green_cfg.get('h_low', 40),
            green_cfg.get('s_low', 100),
            green_cfg.get('v_low', 100)
        ])
        self.green_high = np.array([
            green_cfg.get('h_high', 85),
            green_cfg.get('s_high', 255),
            green_cfg.get('v_high', 255)
        ])

    def _get_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Extract region of interest from frame."""
        height, width = frame.shape[:2]
        y1 = int(height * self.roi_top)
        y2 = int(height * self.roi_bottom)
        x1 = int(width * self.roi_left)
        x2 = int(width * self.roi_right)

        roi = frame[y1:y2, x1:x2]
        return roi, x1, y1

    def _detect_color(self, hsv: np.ndarray,
                      color: str) -> List[DetectedTrafficLight]:
        """Detect traffic lights of a specific color."""
        detections = []

        # Create mask based on color
        if color == 'red':
            mask1 = cv2.inRange(hsv, self.red_low_1, self.red_high_1)
            mask2 = cv2.inRange(hsv, self.red_low_2, self.red_high_2)
            mask = cv2.bitwise_or(mask1, mask2)
            state = TrafficLightState.RED
        elif color == 'yellow':
            mask = cv2.inRange(hsv, self.yellow_low, self.yellow_high)
            state = TrafficLightState.YELLOW
        elif color == 'green':
            mask = cv2.inRange(hsv, self.green_low, self.green_high)
            state = TrafficLightState.GREEN
        else:
            return detections

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by minimum area
            if area < self.min_area:
                continue

            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity < self.min_circularity:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Check aspect ratio (should be roughly square for traffic light)
            aspect_ratio = w / (h + 1e-6)
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            # Calculate confidence based on circularity and area
            confidence = min(1.0, circularity * (area / 1000))

            center = (x + w // 2, y + h // 2)

            detections.append(DetectedTrafficLight(
                state=state,
                bbox=(x, y, w, h),
                center=center,
                confidence=confidence,
                area=area
            ))

        return detections

    def detect(self, frame: np.ndarray) -> Tuple[TrafficLightState, List[DetectedTrafficLight]]:
        """
        Detect traffic lights in frame.

        Args:
            frame: BGR image

        Returns:
            Tuple of (current_state, all_detections)
        """
        if frame is None:
            return TrafficLightState.NONE, []

        # Get ROI
        roi, x_offset, y_offset = self._get_roi(frame)

        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Detect each color
        all_detections = []

        for color in ['red', 'yellow', 'green']:
            detections = self._detect_color(hsv, color)

            # Adjust coordinates to full frame
            for det in detections:
                det.bbox = (
                    det.bbox[0] + x_offset,
                    det.bbox[1] + y_offset,
                    det.bbox[2],
                    det.bbox[3]
                )
                det.center = (
                    det.center[0] + x_offset,
                    det.center[1] + y_offset
                )

            all_detections.extend(detections)

        # Determine current state (largest/most confident detection)
        current_state = TrafficLightState.NONE

        if all_detections:
            # Sort by area (largest first)
            all_detections.sort(key=lambda d: d.area, reverse=True)
            current_state = all_detections[0].state

        # Apply state smoothing
        self.state_history.append(current_state)
        if len(self.state_history) > self.history_size:
            self.state_history.pop(0)

        # Use most common state in history
        smoothed_state = self._get_smoothed_state()

        return smoothed_state, all_detections

    def _get_smoothed_state(self) -> TrafficLightState:
        """Get smoothed state from history."""
        if not self.state_history:
            return TrafficLightState.NONE

        # Count occurrences
        counts = {}
        for state in self.state_history:
            counts[state] = counts.get(state, 0) + 1

        # Return most common (but prioritize non-NONE states)
        non_none_states = {k: v for k, v in counts.items()
                          if k != TrafficLightState.NONE}

        if non_none_states:
            return max(non_none_states, key=non_none_states.get)
        return TrafficLightState.NONE

    def get_action(self, state: TrafficLightState) -> str:
        """
        Get recommended action for traffic light state.

        Args:
            state: Current traffic light state

        Returns:
            Action string: 'stop', 'slow', 'go', or 'continue'
        """
        actions = {
            TrafficLightState.RED: 'stop',
            TrafficLightState.YELLOW: 'slow',
            TrafficLightState.GREEN: 'go',
            TrafficLightState.NONE: 'continue'
        }
        return actions.get(state, 'continue')

    def draw_detections(self, frame: np.ndarray,
                        state: TrafficLightState,
                        detections: List[DetectedTrafficLight]) -> np.ndarray:
        """Draw traffic light detections on frame."""
        output = frame.copy()
        height = frame.shape[0]

        colors = {
            TrafficLightState.RED: (0, 0, 255),
            TrafficLightState.YELLOW: (0, 255, 255),
            TrafficLightState.GREEN: (0, 255, 0),
            TrafficLightState.NONE: (128, 128, 128)
        }

        # Draw ROI rectangle
        h, w = frame.shape[:2]
        roi_pts = [
            (int(w * self.roi_left), int(h * self.roi_top)),
            (int(w * self.roi_right), int(h * self.roi_bottom))
        ]
        cv2.rectangle(output, roi_pts[0], roi_pts[1], (100, 100, 100), 1)

        # Draw detections
        for det in detections:
            x, y, bw, bh = det.bbox
            color = colors.get(det.state, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 2)

            # Draw circle at center
            cv2.circle(output, det.center, 5, color, -1)

            # Label
            label = f"{det.state.value} {det.confidence:.2f}"
            cv2.putText(output, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw current state indicator
        state_color = colors.get(state, (128, 128, 128))
        action = self.get_action(state)

        # Traffic light icon
        icon_x, icon_y = 10, 60
        icon_size = 20

        # Draw traffic light background
        cv2.rectangle(output, (icon_x, icon_y),
                     (icon_x + icon_size, icon_y + icon_size * 3 + 10),
                     (50, 50, 50), -1)

        # Draw lights
        light_colors = [
            ((0, 0, 100) if state != TrafficLightState.RED else (0, 0, 255)),
            ((0, 100, 100) if state != TrafficLightState.YELLOW else (0, 255, 255)),
            ((0, 100, 0) if state != TrafficLightState.GREEN else (0, 255, 0))
        ]

        for i, lc in enumerate(light_colors):
            center = (icon_x + icon_size // 2, icon_y + 10 + i * (icon_size + 2))
            cv2.circle(output, center, icon_size // 2 - 2, lc, -1)

        # Draw action text
        cv2.putText(output, f"TRAFFIC: {state.value.upper()}",
                   (icon_x + icon_size + 10, icon_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        cv2.putText(output, f"Action: {action}",
                   (icon_x + icon_size + 10, icon_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output
