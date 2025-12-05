"""
Traffic Sign and Light Detection Module using YOLOv8.
Detects traffic lights and road signs using fine-tuned YOLO model.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")


class TrafficLightState(Enum):
    """Traffic light states."""
    NONE = "none"
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


class SignType(Enum):
    """Road sign types."""
    STOP = "stop_sign"
    BUMP = "Bump"
    NO_PARKING = "No_Parking"
    NO_STOPPING = "No_Stopping"
    NO_U_TURN = "No_U-Turn"
    PEDESTRIAN = "Pedestrian_sign"
    ROAD_WORK = "Road_Work"
    SPEED_LIMIT_40 = "Speed_Limit_40"
    SPEED_LIMIT_90 = "Speed_Limit_90"
    SPEED_LIMIT_120 = "Speed_Limit_120"


@dataclass
class DetectedTrafficLight:
    """Represents a detected traffic light."""
    state: TrafficLightState
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]
    confidence: float
    area: int


@dataclass
class DetectedSign:
    """Represents a detected road sign."""
    sign_type: SignType
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]
    confidence: float
    area: int


class TrafficSignLightDetector:
    """YOLO-based traffic sign and light detector."""

    def __init__(self, config: dict, mode: str = 'webcam'):
        """
        Initialize traffic sign and light detector.

        Args:
            config: Configuration dictionary from config.yaml
            mode: Operation mode ('webcam', 'video', 'image', 'ros')
                  - 'video': uses traffic_sign_lights.pt
                  - 'webcam'/'ros': uses traffic6.pt
        """
        self.config = config.get('traffic_sign_light', {})

        # Select model based on mode
        if mode == 'video':
            # Use traffic_sign_lights.pt for video input
            self.model_path = self.config.get('model_video', 'models/traffic_sign_lights.pt')
            print(f"  [Traffic Detector] Using video model: {self.model_path}")
        else:
            # Use traffic6.pt for webcam/ros input
            self.model_path = self.config.get('model_webcam', 'models/traffic6.pt')
            print(f"  [Traffic Detector] Using webcam model: {self.model_path}")

        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)

        # Class mapping for traffic6.pt model (11 classes, 0-indexed)
        self.class_names = {
            0: 'green_traffic',
            1: 'red_traffic',
            2: 'Bump',
            3: 'No_Parking',
            4: 'No_Stopping',
            5: 'No_U-Turn',
            6: 'Pedestrian_sign',
            7: 'Road_Work',
            8: 'Speed_Limit_120',
            9: 'Speed_Limit_40',
            10: 'Speed_Limit_90'
        }

        # Traffic light class IDs
        self.traffic_light_classes = {
            0: TrafficLightState.GREEN,
            1: TrafficLightState.RED
        }

        # Road sign class IDs
        self.sign_classes = {
            2: SignType.BUMP,
            3: SignType.NO_PARKING,
            4: SignType.NO_STOPPING,
            5: SignType.NO_U_TURN,
            6: SignType.PEDESTRIAN,
            7: SignType.ROAD_WORK,
            8: SignType.SPEED_LIMIT_120,
            9: SignType.SPEED_LIMIT_40,
            10: SignType.SPEED_LIMIT_90
        }

        # State tracking for traffic lights
        self.light_state_history = []
        self.history_size = 5

        self.model = None
        if YOLO_AVAILABLE:
            self._load_model()

    def _load_model(self):
        """Load fine-tuned YOLO model."""
        try:
            self.model = YOLO(self.model_path)
            print(f"[TrafficSignLightDetector] Loaded model: {self.model_path}")
        except Exception as e:
            print(f"[TrafficSignLightDetector] Error loading model: {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> Tuple[TrafficLightState, List[DetectedTrafficLight], List[DetectedSign]]:
        """
        Detect traffic lights and signs in frame.

        Args:
            frame: BGR image

        Returns:
            Tuple of (traffic_light_state, traffic_light_detections, sign_detections)
        """
        if self.model is None or frame is None:
            return TrafficLightState.NONE, [], []

        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width

        traffic_lights = []
        signs = []

        # Run YOLO detection
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)

                # Check if it's a traffic light
                if cls_id in self.traffic_light_classes:
                    light = DetectedTrafficLight(
                        state=self.traffic_light_classes[cls_id],
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y),
                        confidence=float(conf),
                        area=area
                    )
                    traffic_lights.append(light)

                # Check if it's a road sign
                elif cls_id in self.sign_classes:
                    sign = DetectedSign(
                        sign_type=self.sign_classes[cls_id],
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y),
                        confidence=float(conf),
                        area=area
                    )
                    signs.append(sign)

        # Determine overall traffic light state (prioritize red > yellow > green)
        light_state = self._get_stable_light_state(traffic_lights)

        return light_state, traffic_lights, signs

    def _get_stable_light_state(self, traffic_lights: List[DetectedTrafficLight]) -> TrafficLightState:
        """
        Get stable traffic light state using history.

        Args:
            traffic_lights: List of detected traffic lights

        Returns:
            Most stable traffic light state
        """
        if not traffic_lights:
            current_state = TrafficLightState.NONE
        else:
            # Priority: RED > YELLOW > GREEN
            if any(light.state == TrafficLightState.RED for light in traffic_lights):
                current_state = TrafficLightState.RED
            elif any(light.state == TrafficLightState.YELLOW for light in traffic_lights):
                current_state = TrafficLightState.YELLOW
            elif any(light.state == TrafficLightState.GREEN for light in traffic_lights):
                current_state = TrafficLightState.GREEN
            else:
                current_state = TrafficLightState.NONE

        # Add to history
        self.light_state_history.append(current_state)
        if len(self.light_state_history) > self.history_size:
            self.light_state_history.pop(0)

        # Return most common state in history
        if not self.light_state_history:
            return TrafficLightState.NONE

        # Count occurrences
        state_counts = {}
        for state in self.light_state_history:
            state_counts[state] = state_counts.get(state, 0) + 1

        # Return most common (with priority for safety)
        if TrafficLightState.RED in state_counts and state_counts[TrafficLightState.RED] >= 2:
            return TrafficLightState.RED
        elif TrafficLightState.YELLOW in state_counts and state_counts[TrafficLightState.YELLOW] >= 2:
            return TrafficLightState.YELLOW

        return max(state_counts, key=state_counts.get)

    def draw_detections(self, frame: np.ndarray, light_state: TrafficLightState,
                       traffic_lights: List[DetectedTrafficLight],
                       signs: List[DetectedSign]) -> np.ndarray:
        """
        Draw detection visualizations on frame.

        Args:
            frame: Input image
            light_state: Current traffic light state
            traffic_lights: List of detected traffic lights
            signs: List of detected road signs

        Returns:
            Annotated image
        """
        output = frame.copy()

        # Draw traffic lights
        for light in traffic_lights:
            x1, y1, x2, y2 = light.bbox

            # Color based on light state
            if light.state == TrafficLightState.RED:
                color = (0, 0, 255)
            elif light.state == TrafficLightState.YELLOW:
                color = (0, 255, 255)
            elif light.state == TrafficLightState.GREEN:
                color = (0, 255, 0)
            else:
                color = (128, 128, 128)

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{light.state.value.upper()} {light.confidence:.2f}"
            cv2.putText(output, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw road signs
        for sign in signs:
            x1, y1, x2, y2 = sign.bbox

            # Color for signs (blue)
            color = (255, 0, 0)

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{sign.sign_type.value} {sign.confidence:.2f}"
            cv2.putText(output, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw overall state indicator
        state_color = {
            TrafficLightState.RED: (0, 0, 255),
            TrafficLightState.YELLOW: (0, 255, 255),
            TrafficLightState.GREEN: (0, 255, 0),
            TrafficLightState.NONE: (128, 128, 128)
        }

        color = state_color.get(light_state, (128, 128, 128))
        cv2.putText(output, f"Traffic: {light_state.value.upper()}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw detected signs summary
        if signs:
            sign_text = f"Signs: {len(signs)}"
            cv2.putText(output, sign_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return output
