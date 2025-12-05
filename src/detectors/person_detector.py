"""
Person Detection Module using YOLOv8.
Detects pedestrians for autonomous driving safety.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")


@dataclass
class DetectedPerson:
    """Represents a detected person."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    center: Tuple[int, int]
    area: int
    relative_size: float  # Size relative to frame (0-1)


class PersonDetector:
    """YOLOv8-based person detector for TurtleBot navigation."""

    def __init__(self, config: dict):
        """
        Initialize person detector.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config.get('person_detection', {})
        self.model_name = self.config.get('model', 'models/yolov8n.pt')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.person_class_id = self.config.get('person_class_id', 0)
        self.danger_zone_ratio = self.config.get('danger_zone_ratio', 0.4)
        self.caution_zone_ratio = self.config.get('caution_zone_ratio', 0.25)

        self.model = None
        if YOLO_AVAILABLE:
            self._load_model()

    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            self.model = YOLO(self.model_name)
            print(f"[PersonDetector] Loaded model: {self.model_name}")
        except Exception as e:
            print(f"[PersonDetector] Error loading model: {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[DetectedPerson]:
        """
        Detect people in frame.

        Args:
            frame: BGR image

        Returns:
            List of DetectedPerson objects sorted by size (closest first)
        """
        if self.model is None or frame is None:
            return []

        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width
        detections = []

        results = self.model(frame, verbose=False, conf=self.confidence_threshold)

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                if int(box.cls[0]) != self.person_class_id:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1
                area = width * height
                relative_size = area / frame_area

                detections.append(DetectedPerson(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    center=(center_x, center_y),
                    area=area,
                    relative_size=relative_size
                ))

        # Sort by relative size (largest/closest first)
        detections.sort(key=lambda p: p.relative_size, reverse=True)
        return detections

    def get_danger_level(self, detections: List[DetectedPerson],
                         frame_width: int) -> Tuple[str, Optional[DetectedPerson], Optional[str]]:
        """
        Assess danger level from detected persons.

        Args:
            detections: List of detected persons
            frame_width: Width of frame for position analysis

        Returns:
            Tuple of (danger_level, closest_person, avoidance_direction)
            danger_level: 'safe' or 'stop'
            avoidance_direction: Always None (no avoidance)
        """
        if not detections:
            return 'safe', None, None

        # If ANY person is detected, STOP immediately
        closest = detections[0]

        return 'stop', closest, None

    def draw_detections(self, frame: np.ndarray,
                        detections: List[DetectedPerson],
                        danger_level: str = 'safe') -> np.ndarray:
        """Draw detection boxes on frame."""
        output = frame.copy()

        colors = {
            'safe': (0, 255, 0),
            'caution': (0, 165, 255),
            'danger': (0, 0, 255),
            'stop': (0, 0, 255)
        }

        for i, person in enumerate(detections):
            x1, y1, x2, y2 = person.bbox
            color = colors.get(danger_level, (255, 255, 255))

            # Draw box
            thickness = 3 if i == 0 else 2
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"Person {person.confidence:.2f}"
            size_label = f"Size: {person.relative_size:.2%}"

            cv2.putText(output, label, (x1, y1 - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(output, size_label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw center point
            cv2.circle(output, person.center, 5, color, -1)

        # Draw danger level indicator
        if detections:
            status_color = colors.get(danger_level, (255, 255, 255))
            cv2.putText(output, f"PEDESTRIAN: {danger_level.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        return output
