#!/usr/bin/env python3
"""
Quick test script for traffic sign and light detection
"""
import cv2
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.detectors.traffic_sign_light_detector import TrafficSignLightDetector

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize detector
print("Loading traffic sign and light detector...")
detector = TrafficSignLightDetector(config)

# Open video
video_path = 'road_test.MOV'
print(f"Opening video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    sys.exit(1)

frame_count = 0
processed_count = 0

while cap.isOpened() and processed_count < 100:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process every 10th frame
    if frame_count % 10 == 0:
        print(f"\nProcessing frame {frame_count}...")

        # Detect
        traffic_state, traffic_lights, signs = detector.detect(frame)

        print(f"  Traffic Light State: {traffic_state.value}")
        print(f"  Traffic Lights Detected: {len(traffic_lights)}")
        if traffic_lights:
            for light in traffic_lights:
                print(f"    - {light.state.value}: confidence={light.confidence:.2f}")

        print(f"  Road Signs Detected: {len(signs)}")
        if signs:
            for sign in signs:
                print(f"    - {sign.sign_type.value}: confidence={sign.confidence:.2f}")

        # Draw detections
        output = detector.draw_detections(frame, traffic_state, traffic_lights, signs)

        # Save annotated frame
        output_path = f"output_frame_{frame_count}.jpg"
        cv2.imwrite(output_path, output)
        print(f"  Saved: {output_path}")

        processed_count += 1

cap.release()
print(f"\nProcessed {processed_count} frames from {frame_count} total frames")
print("Test complete!")
