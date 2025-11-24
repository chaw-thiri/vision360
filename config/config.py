"""
Configuration settings for the autonomous driving vision system.
"""

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Person detection settings
PERSON_MODEL = "yolov8n.pt"  # YOLOv8 nano model (fast)
PERSON_CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0  # COCO class ID for person

# Lane detection settings
LANE_ROI_TOP_RATIO = 0.6  # Region of interest starts at 60% from top
LANE_ROI_BOTTOM_RATIO = 1.0
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
HOUGH_RHO = 2
HOUGH_THETA_DEGREES = 1
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 40
HOUGH_MAX_LINE_GAP = 100

# Lane color detection (HSV ranges for white and yellow lanes)
WHITE_LANE_HSV_LOW = (0, 0, 200)
WHITE_LANE_HSV_HIGH = (180, 30, 255)
YELLOW_LANE_HSV_LOW = (15, 80, 100)
YELLOW_LANE_HSV_HIGH = (35, 255, 255)

# Decision making settings
SAFE_DISTANCE_PIXELS = 150  # Minimum safe distance from detected person
LANE_CENTER_TOLERANCE = 50  # Tolerance for lane centering (pixels)
STEERING_SENSITIVITY = 0.01  # Steering adjustment sensitivity

# TurtleBot control settings
MAX_LINEAR_VELOCITY = 0.22  # m/s (TurtleBot3 max)
MAX_ANGULAR_VELOCITY = 2.84  # rad/s (TurtleBot3 max)
STOP_DISTANCE = 0.5  # meters - stop if person is within this distance

# Debug/visualization settings
SHOW_DEBUG_WINDOWS = True
SAVE_DEBUG_FRAMES = False
DEBUG_OUTPUT_DIR = "debug_output"
