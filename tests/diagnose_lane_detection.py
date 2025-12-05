#!/usr/bin/env python3
"""
Diagnostic script to analyze lane detection issues on real road video
"""
import cv2
import numpy as np
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.detectors.lane_detector import LaneDetector

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize detector
print("Loading lane detector...")
detector = LaneDetector(config)

# Open video
video_path = 'road_test.MOV'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video")
    sys.exit(1)

# Sample frames at different points
test_frames = [100, 300, 500, 700, 900]

for frame_num in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        continue

    print(f"\n{'='*60}")
    print(f"Frame {frame_num}")
    print(f"{'='*60}")

    # Get frame info
    height, width = frame.shape[:2]
    print(f"Frame size: {width}x{height}")

    # Detect lanes
    lane_info = detector.detect(frame)

    print(f"Lane detected: {lane_info.lane_detected}")
    print(f"Lane center offset: {lane_info.lane_center_offset:.3f}")
    print(f"Steering angle: {lane_info.steering_angle:.3f}")
    print(f"Confidence: {lane_info.confidence:.3f}")
    print(f"Left lines: {len(lane_info.left_lines)}")
    print(f"Right lines: {len(lane_info.right_lines)}")

    # Create diagnostic visualization
    diagnostic = frame.copy()

    # Draw ROI
    roi_top = int(height * 0.5)
    roi_bottom = height
    cv2.rectangle(diagnostic, (0, roi_top), (width, roi_bottom), (255, 0, 0), 2)
    cv2.putText(diagnostic, "ROI", (10, roi_top + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Extract ROI and convert to HSV
    roi = frame[roi_top:roi_bottom, :]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Test different color ranges
    # White lane markings (typical on roads)
    white_mask = cv2.inRange(hsv_roi,
                             np.array([0, 0, 180]),
                             np.array([180, 35, 255]))

    # Yellow lane markings
    yellow_mask = cv2.inRange(hsv_roi,
                              np.array([20, 100, 100]),
                              np.array([35, 255, 255]))

    # Combined mask
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Show mask statistics
    white_pixels = np.count_nonzero(white_mask)
    yellow_pixels = np.count_nonzero(yellow_mask)
    total_pixels = roi.shape[0] * roi.shape[1]

    print(f"White pixels in ROI: {white_pixels} ({100*white_pixels/total_pixels:.1f}%)")
    print(f"Yellow pixels in ROI: {yellow_pixels} ({100*yellow_pixels/total_pixels:.1f}%)")

    # Draw detected lanes
    output = detector.draw_lanes(diagnostic, lane_info)

    # Save diagnostic images
    cv2.imwrite(f'diag_frame_{frame_num}.jpg', output)
    cv2.imwrite(f'diag_mask_{frame_num}.jpg', combined_mask)
    print(f"Saved: diag_frame_{frame_num}.jpg and diag_mask_{frame_num}.jpg")

cap.release()
print("\nDiagnostic complete!")
