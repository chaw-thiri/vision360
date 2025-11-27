"""
Boundary Platform Detection Module.
Detects black boundary platforms and finds navigable openings.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class BoundaryInfo:
    """Information about detected boundary platforms."""
    left_blocked: bool
    center_blocked: bool
    right_blocked: bool
    left_coverage: float  # 0.0 to 1.0
    center_coverage: float
    right_coverage: float
    all_blocked: bool
    suggested_direction: str  # 'left', 'right', 'center', 'blocked'
    avoidance_angle: float  # Steering adjustment (-1.0 to 1.0)


class BoundaryPlatformDetector:
    """Detects black boundary platforms and determines navigation strategy."""

    def __init__(self, config: dict):
        """
        Initialize boundary platform detector.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config.get('boundary_detection', {})

        # Check if boundary detection is enabled
        self.enabled = self.config.get('enabled', True)

        # ROI settings - bottom portion of frame where boundaries appear
        roi = self.config.get('roi', {})
        self.roi_top = roi.get('top_ratio', 0.6)
        self.roi_bottom = roi.get('bottom_ratio', 1.0)

        # Black platform color range (HSV)
        black_platform = self.config.get('black_platform', {})
        self.black_low = np.array([
            black_platform.get('h_low', 0),
            black_platform.get('s_low', 0),
            black_platform.get('v_low', 0)
        ])
        self.black_high = np.array([
            black_platform.get('h_high', 180),
            black_platform.get('s_high', 255),
            black_platform.get('v_high', 50)
        ])

        # Zone division (divide frame into left/center/right)
        self.left_boundary = self.config.get('left_zone_boundary', 0.33)
        self.right_boundary = self.config.get('right_zone_boundary', 0.67)

        # Coverage thresholds
        self.block_threshold = self.config.get('block_threshold', 0.3)  # 30% coverage = blocked
        self.full_block_threshold = self.config.get('full_block_threshold', 0.8)  # 80% = completely blocked

        # Avoidance parameters
        self.avoidance_strength = self.config.get('avoidance_strength', 0.6)

    def _get_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Extract region of interest from frame."""
        height = frame.shape[0]
        roi_top_y = int(height * self.roi_top)
        roi = frame[roi_top_y:, :]
        return roi, roi_top_y

    def _detect_black_platforms(self, frame: np.ndarray) -> np.ndarray:
        """Detect black platforms using HSV thresholding."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect black regions
        black_mask = cv2.inRange(hsv, self.black_low, self.black_high)

        # Clean up noise with morphological operations
        kernel = np.ones((7, 7), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)

        return black_mask

    def _analyze_zones(self, black_mask: np.ndarray) -> Tuple[float, float, float]:
        """
        Analyze black platform coverage in left, center, and right zones.

        Args:
            black_mask: Binary mask of black regions

        Returns:
            Tuple of (left_coverage, center_coverage, right_coverage)
        """
        height, width = black_mask.shape

        # Define zone boundaries
        left_end = int(width * self.left_boundary)
        right_start = int(width * self.right_boundary)

        # Extract zones
        left_zone = black_mask[:, :left_end]
        center_zone = black_mask[:, left_end:right_start]
        right_zone = black_mask[:, right_start:]

        # Calculate coverage (percentage of white pixels in each zone)
        left_coverage = np.sum(left_zone > 0) / (left_zone.size + 1e-6)
        center_coverage = np.sum(center_zone > 0) / (center_zone.size + 1e-6)
        right_coverage = np.sum(right_zone > 0) / (right_zone.size + 1e-6)

        return left_coverage, center_coverage, right_coverage

    def _determine_navigation(self, left_cov: float, center_cov: float,
                             right_cov: float) -> Tuple[str, float]:
        """
        Determine navigation direction based on zone coverage.

        Args:
            left_cov: Left zone coverage (0.0 to 1.0)
            center_cov: Center zone coverage (0.0 to 1.0)
            right_cov: Right zone coverage (0.0 to 1.0)

        Returns:
            Tuple of (direction: str, avoidance_angle: float)
        """
        # Check if zones are blocked
        left_blocked = left_cov > self.block_threshold
        center_blocked = center_cov > self.block_threshold
        right_blocked = right_cov > self.block_threshold

        # Check if completely blocked
        all_blocked = (left_cov > self.full_block_threshold and
                      center_cov > self.full_block_threshold and
                      right_cov > self.full_block_threshold)

        if all_blocked:
            return 'blocked', 0.0

        # Calculate clearance scores (inverse of coverage)
        left_clearance = 1.0 - left_cov
        center_clearance = 1.0 - center_cov
        right_clearance = 1.0 - right_cov

        # Determine best direction
        if not center_blocked:
            # Center is clear - prefer staying centered
            if left_blocked and not right_blocked:
                # Left blocked, steer right
                return 'right', self.avoidance_strength * (left_cov - right_cov)
            elif right_blocked and not left_blocked:
                # Right blocked, steer left
                return 'left', -self.avoidance_strength * (right_cov - left_cov)
            else:
                # Center clear, minor adjustments to stay centered
                balance = (left_cov - right_cov) * 0.3
                if abs(balance) > 0.1:
                    direction = 'right' if balance > 0 else 'left'
                    return direction, balance
                return 'center', 0.0
        else:
            # Center is blocked - must choose left or right
            if left_clearance > right_clearance:
                # Left is more clear
                if left_blocked:
                    # Both sides have obstacles, but left is better
                    return 'left', -self.avoidance_strength * 0.5
                else:
                    # Left is clear
                    return 'left', -self.avoidance_strength
            else:
                # Right is more clear
                if right_blocked:
                    # Both sides have obstacles, but right is better
                    return 'right', self.avoidance_strength * 0.5
                else:
                    # Right is clear
                    return 'right', self.avoidance_strength

    def detect(self, frame: np.ndarray) -> BoundaryInfo:
        """
        Detect boundary platforms and determine navigation strategy.

        Args:
            frame: BGR image

        Returns:
            BoundaryInfo object with detection results
        """
        if frame is None or not self.enabled:
            return BoundaryInfo(
                left_blocked=False,
                center_blocked=False,
                right_blocked=False,
                left_coverage=0.0,
                center_coverage=0.0,
                right_coverage=0.0,
                all_blocked=False,
                suggested_direction='center',
                avoidance_angle=0.0
            )

        # Get ROI (bottom portion of frame)
        roi, roi_offset = self._get_roi(frame)

        # Detect black platforms
        black_mask = self._detect_black_platforms(roi)

        # Analyze zones
        left_cov, center_cov, right_cov = self._analyze_zones(black_mask)

        # Determine navigation strategy
        direction, avoidance_angle = self._determine_navigation(
            left_cov, center_cov, right_cov
        )

        # Check blocking status
        left_blocked = left_cov > self.block_threshold
        center_blocked = center_cov > self.block_threshold
        right_blocked = right_cov > self.block_threshold
        all_blocked = direction == 'blocked'

        return BoundaryInfo(
            left_blocked=left_blocked,
            center_blocked=center_blocked,
            right_blocked=right_blocked,
            left_coverage=left_cov,
            center_coverage=center_cov,
            right_coverage=right_cov,
            all_blocked=all_blocked,
            suggested_direction=direction,
            avoidance_angle=avoidance_angle
        )

    def draw_detection(self, frame: np.ndarray, boundary_info: BoundaryInfo) -> np.ndarray:
        """
        Draw boundary detection visualization on frame.

        Args:
            frame: BGR image
            boundary_info: Boundary detection results

        Returns:
            Frame with visualization overlay
        """
        output = frame.copy()
        height, width = frame.shape[:2]

        # Get ROI bounds
        roi_top_y = int(height * self.roi_top)
        left_x = int(width * self.left_boundary)
        right_x = int(width * self.right_boundary)

        # Draw ROI rectangle
        cv2.rectangle(output, (0, roi_top_y), (width, height), (100, 100, 100), 1)

        # Draw zone dividers
        cv2.line(output, (left_x, roi_top_y), (left_x, height), (150, 150, 150), 1)
        cv2.line(output, (right_x, roi_top_y), (right_x, height), (150, 150, 150), 1)

        # Color zones based on blocking status
        overlay = output.copy()
        zone_height = height - roi_top_y

        # Left zone
        left_color = (0, 0, 150) if boundary_info.left_blocked else (0, 150, 0)
        cv2.rectangle(overlay, (0, roi_top_y), (left_x, height), left_color, -1)

        # Center zone
        center_color = (0, 0, 150) if boundary_info.center_blocked else (0, 150, 0)
        cv2.rectangle(overlay, (left_x, roi_top_y), (right_x, height), center_color, -1)

        # Right zone
        right_color = (0, 0, 150) if boundary_info.right_blocked else (0, 150, 0)
        cv2.rectangle(overlay, (right_x, roi_top_y), (width, height), right_color, -1)

        # Blend overlay
        output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)

        # Draw coverage percentages
        cv2.putText(output, f"L:{boundary_info.left_coverage*100:.0f}%",
                   (10, roi_top_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1)
        cv2.putText(output, f"C:{boundary_info.center_coverage*100:.0f}%",
                   (left_x + 20, roi_top_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1)
        cv2.putText(output, f"R:{boundary_info.right_coverage*100:.0f}%",
                   (right_x + 10, roi_top_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1)

        # Draw navigation direction
        if boundary_info.all_blocked:
            color = (0, 0, 255)  # Red
            message = "ALL BLOCKED - STOPPING"
        elif boundary_info.suggested_direction == 'left':
            color = (255, 0, 255)  # Magenta
            message = "Navigate LEFT"
        elif boundary_info.suggested_direction == 'right':
            color = (255, 255, 0)  # Cyan
            message = "Navigate RIGHT"
        else:
            color = (0, 255, 0)  # Green
            message = "Path CLEAR"

        cv2.putText(output, f"BOUNDARY: {message}",
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw avoidance arrow
        if not boundary_info.all_blocked and abs(boundary_info.avoidance_angle) > 0.1:
            center_x = width // 2
            arrow_y = height - 30
            arrow_length = int(abs(boundary_info.avoidance_angle) * 100)
            arrow_end_x = center_x + int(boundary_info.avoidance_angle * 100)
            cv2.arrowedLine(output, (center_x, arrow_y), (arrow_end_x, arrow_y),
                          color, 3, tipLength=0.3)

        return output

    def get_debug_view(self, frame: np.ndarray) -> np.ndarray:
        """Get debug visualization showing black mask."""
        roi, _ = self._get_roi(frame)

        # Get black mask
        black_mask = self._detect_black_platforms(roi)

        # Convert to BGR for display
        black_bgr = cv2.cvtColor(black_mask, cv2.COLOR_GRAY2BGR)

        # Draw zone dividers
        height, width = black_mask.shape
        left_x = int(width * self.left_boundary)
        right_x = int(width * self.right_boundary)

        cv2.line(black_bgr, (left_x, 0), (left_x, height), (0, 255, 0), 2)
        cv2.line(black_bgr, (right_x, 0), (right_x, height), (0, 255, 0), 2)

        # Add label
        cv2.putText(black_bgr, "Black Boundary Platforms", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return black_bgr
