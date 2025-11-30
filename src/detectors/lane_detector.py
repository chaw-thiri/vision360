"""
Lane Detection Module for Indoor Track.
Detects taped lane lines for autonomous navigation.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class LaneInfo:
    """Information about detected lanes."""
    left_lane: Optional[np.ndarray]  # Line points
    right_lane: Optional[np.ndarray]
    center_offset: float  # Offset from lane center (negative=left, positive=right)
    steering_angle: float  # Suggested steering angle
    lane_detected: bool
    confidence: float


class LaneDetector:
    """Detects lane lines for indoor taped tracks."""

    def __init__(self, config: dict):
        """
        Initialize lane detector.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config.get('lane_detection', {})

        # ROI settings
        roi = self.config.get('roi', {})
        self.roi_top = roi.get('top_ratio', 0.5)
        self.roi_bottom = roi.get('bottom_ratio', 1.0)

        # Color ranges for tape detection
        self._setup_color_ranges()

        # Edge detection
        canny = self.config.get('canny', {})
        self.canny_low = canny.get('low_threshold', 50)
        self.canny_high = canny.get('high_threshold', 150)

        # Hough transform
        hough = self.config.get('hough', {})
        self.hough_rho = hough.get('rho', 2)
        self.hough_theta = np.pi / 180 * hough.get('theta_degrees', 1)
        self.hough_threshold = hough.get('threshold', 50)
        self.hough_min_line = hough.get('min_line_length', 40)
        self.hough_max_gap = hough.get('max_line_gap', 100)

        # Sliding window parameters
        self.num_windows = 9
        self.window_margin = 50
        self.min_pixels = 50

    def _setup_color_ranges(self):
        """Setup HSV color ranges for tape detection."""
        # Black lines (for vertical lane boundaries)
        black = self.config.get('black_line', {})
        self.black_low = np.array([
            black.get('h_low', 0),
            black.get('s_low', 0),
            black.get('v_low', 0)
        ])
        self.black_high = np.array([
            black.get('h_high', 180),
            black.get('s_high', 255),
            black.get('v_high', 60)
        ])

        # White tape (backup)
        white = self.config.get('white_tape', {})
        self.white_low = np.array([
            white.get('h_low', 0),
            white.get('s_low', 0),
            white.get('v_low', 200)
        ])
        self.white_high = np.array([
            white.get('h_high', 180),
            white.get('s_high', 30),
            white.get('v_high', 255)
        ])

        # Yellow tape (backup)
        yellow = self.config.get('yellow_tape', {})
        self.yellow_low = np.array([
            yellow.get('h_low', 20),
            yellow.get('s_low', 100),
            yellow.get('v_low', 100)
        ])
        self.yellow_high = np.array([
            yellow.get('h_high', 35),
            yellow.get('s_high', 255),
            yellow.get('v_high', 255)
        ])

        # Blue tape (backup)
        blue = self.config.get('blue_tape', {})
        self.blue_low = np.array([
            blue.get('h_low', 100),
            blue.get('s_low', 100),
            blue.get('v_low', 100)
        ])
        self.blue_high = np.array([
            blue.get('h_high', 130),
            blue.get('s_high', 255),
            blue.get('v_high', 255)
        ])

    def _get_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Extract region of interest from frame."""
        height = frame.shape[0]
        roi_top_y = int(height * self.roi_top)
        roi = frame[roi_top_y:, :]
        return roi, roi_top_y

    def _detect_tape_colors(self, frame: np.ndarray) -> np.ndarray:
        """Detect black lane only."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect BLACK lane ONLY (primary and only)
        black_mask = cv2.inRange(hsv, self.black_low, self.black_high)

        # Clean up noise with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)

        return black_mask

    def _detect_edges(self, frame: np.ndarray) -> np.ndarray:
        """Detect edges using Canny edge detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        return edges

    def _detect_lines(self, binary: np.ndarray) -> Optional[np.ndarray]:
        """Detect lines using Hough transform."""
        lines = cv2.HoughLinesP(
            binary,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            minLineLength=self.hough_min_line,
            maxLineGap=self.hough_max_gap
        )
        return lines

    def _separate_lanes(self, lines: np.ndarray,
                        frame_width: int) -> Tuple[List, List]:
        """For black lane following, just collect all valid lines as center lane."""
        center_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Skip horizontal lines (artifacts)
            if abs(y2 - y1) < 10:
                continue

            # Calculate slope
            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 1:
                continue

            slope = dy / dx

            # Accept lines with any reasonable slope (curved lane)
            if abs(slope) < 0.2 or abs(slope) > 10.0:
                continue

            center_lines.append(line[0])

        # Return all lines as "left" lane (we'll treat it as center)
        return center_lines, []

    def _average_lane(self, lines: List, frame_height: int) -> Optional[np.ndarray]:
        """Average multiple lines into single lane line."""
        if not lines:
            return None

        x_coords = []
        y_coords = []

        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        if len(x_coords) < 2:
            return None

        # Fit polynomial (x as function of y for lane lines)
        try:
            poly = np.polyfit(y_coords, x_coords, 1)

            # Draw line from bottom to middle of ROI
            y_start = frame_height
            y_end = int(frame_height * 0.6)

            x_start = int(np.polyval(poly, y_start))
            x_end = int(np.polyval(poly, y_end))

            return np.array([[x_start, y_start, x_end, y_end]])
        except:
            return None

    def _calculate_steering(self, left_lane: Optional[np.ndarray],
                           right_lane: Optional[np.ndarray],
                           frame_width: int,
                           frame_height: int) -> Tuple[float, float]:
        """Calculate steering to follow black lane curvature."""
        center_x = frame_width // 2
        bottom_y = frame_height
        mid_y = int(frame_height * 0.7)  # Look ahead point

        # For black lane following, left_lane contains the detected lane
        if left_lane is not None:
            x1, y1, x2, y2 = left_lane[0]

            # Calculate where the lane is at the look-ahead point
            if y1 != y2:
                slope = (x2 - x1) / (y2 - y1 + 1e-6)
                # Get x position at look-ahead point
                lane_x = int(x1 + slope * (mid_y - y1))
            else:
                lane_x = x1

            # Calculate offset from center
            offset = (lane_x - center_x) / center_x  # Normalized -1 to 1

            # Calculate steering angle to follow the lane
            steering = -offset * 1.0  # Increased steering response for curves
        else:
            # No lane detected
            offset = 0.0
            steering = 0.0

        return offset, steering

    def detect(self, frame: np.ndarray) -> LaneInfo:
        """
        Detect lanes in frame.

        Args:
            frame: BGR image

        Returns:
            LaneInfo object with detection results
        """
        if frame is None:
            return LaneInfo(None, None, 0.0, 0.0, False, 0.0)

        height, width = frame.shape[:2]

        # Get ROI
        roi, roi_offset = self._get_roi(frame)
        roi_height = roi.shape[0]

        # Detect tape using color
        color_mask = self._detect_tape_colors(roi)

        # Also use edge detection
        edges = self._detect_edges(roi)

        # Combine color and edge detection
        combined = cv2.bitwise_or(color_mask, edges)

        # Detect lines
        lines = self._detect_lines(combined)

        if lines is None or len(lines) == 0:
            return LaneInfo(None, None, 0.0, 0.0, False, 0.0)

        # Separate into left and right lanes
        left_lines, right_lines = self._separate_lanes(lines, width)

        # Average the lines
        left_lane = self._average_lane(left_lines, roi_height)
        right_lane = self._average_lane(right_lines, roi_height)

        # Adjust coordinates back to full frame
        if left_lane is not None:
            left_lane[0][1] += roi_offset
            left_lane[0][3] += roi_offset

        if right_lane is not None:
            right_lane[0][1] += roi_offset
            right_lane[0][3] += roi_offset

        # Calculate steering
        offset, steering = self._calculate_steering(
            left_lane, right_lane, width, height
        )

        # Calculate confidence
        confidence = 0.0
        if left_lane is not None:
            confidence += 0.5
        if right_lane is not None:
            confidence += 0.5

        lane_detected = left_lane is not None or right_lane is not None

        return LaneInfo(
            left_lane=left_lane,
            right_lane=right_lane,
            center_offset=offset,
            steering_angle=steering,
            lane_detected=lane_detected,
            confidence=confidence
        )

    def draw_lanes(self, frame: np.ndarray, lane_info: LaneInfo) -> np.ndarray:
        """Draw detected lanes on frame."""
        output = frame.copy()
        height, width = frame.shape[:2]

        # Draw lane overlay
        overlay = np.zeros_like(frame)

        # Draw left lane
        if lane_info.left_lane is not None:
            x1, y1, x2, y2 = lane_info.left_lane[0]
            cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Draw right lane
        if lane_info.right_lane is not None:
            x1, y1, x2, y2 = lane_info.right_lane[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Draw lane area if both lanes detected
        if lane_info.left_lane is not None and lane_info.right_lane is not None:
            pts = np.array([
                [lane_info.left_lane[0][0], lane_info.left_lane[0][1]],
                [lane_info.left_lane[0][2], lane_info.left_lane[0][3]],
                [lane_info.right_lane[0][2], lane_info.right_lane[0][3]],
                [lane_info.right_lane[0][0], lane_info.right_lane[0][1]]
            ], np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            output = cv2.addWeighted(output, 1, overlay, 0.3, 0)

        # Draw center line and steering indicator
        center_x = width // 2
        target_x = int(center_x + lane_info.center_offset * center_x)

        # Vertical center line
        cv2.line(output, (center_x, height), (center_x, height - 100),
                (255, 255, 255), 2)

        # Target position
        cv2.circle(output, (target_x, height - 50), 10, (0, 255, 255), -1)

        # Draw steering info
        status = "LANE DETECTED" if lane_info.lane_detected else "NO LANE"
        color = (0, 255, 0) if lane_info.lane_detected else (0, 0, 255)

        cv2.putText(output, f"{status} ({lane_info.confidence:.0%})",
                   (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        offset_text = f"Offset: {lane_info.center_offset:+.2f}"
        steer_text = f"Steer: {lane_info.steering_angle:+.2f}"
        cv2.putText(output, offset_text, (10, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output, steer_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output

    def get_debug_view(self, frame: np.ndarray) -> np.ndarray:
        """Get debug visualization showing processing steps."""
        roi, _ = self._get_roi(frame)

        color_mask = self._detect_tape_colors(roi)
        edges = self._detect_edges(roi)
        combined = cv2.bitwise_or(color_mask, edges)

        # Convert to BGR for display
        color_bgr = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

        # Stack horizontally
        debug = np.hstack([color_bgr, edges_bgr, combined_bgr])

        # Add labels
        cv2.putText(debug, "Color", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(debug, "Edges", (color_bgr.shape[1] + 10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(debug, "Combined", (color_bgr.shape[1] * 2 + 10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return debug
