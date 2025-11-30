"""
Pydantic models for Vision360 web interface
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class Vehicle(BaseModel):
    """Vehicle data model for fleet management"""
    id: str = Field(..., description="Vehicle ID (e.g., TB3-001)")
    model: str = Field(default="TurtleBot3 Burger", description="Vehicle model")
    product_tag: str = Field(..., description="Product being carried")
    departure: str = Field(..., description="Departure location")
    destination: str = Field(..., description="Destination location")
    latitude: float = Field(..., description="Current latitude")
    longitude: float = Field(..., description="Current longitude")
    battery_percent: int = Field(..., ge=0, le=100, description="Battery percentage")
    status: Literal["active", "idle", "charging"] = Field(..., description="Vehicle status")
    is_real: bool = Field(default=False, description="True if real robot, False if simulated")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "TB3-001",
                "model": "TurtleBot3 Burger",
                "product_tag": "Electronics",
                "departure": "Warehouse A",
                "destination": "Distribution Center 2",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "battery_percent": 85,
                "status": "active",
                "is_real": True,
                "last_updated": "2025-11-30T10:30:00"
            }
        }


class VehicleCreate(BaseModel):
    """Model for creating new vehicles"""
    id: str
    model: str = "TurtleBot3 Burger"
    product_tag: str
    departure: str
    destination: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    battery_percent: int = 100
    status: Literal["active", "idle", "charging"] = "idle"


class VehicleUpdate(BaseModel):
    """Model for updating vehicle information"""
    model: Optional[str] = None
    product_tag: Optional[str] = None
    departure: Optional[str] = None
    destination: Optional[str] = None
    status: Optional[Literal["active", "idle", "charging"]] = None


class RobotStatus(BaseModel):
    """Robot autonomous status from ROS topic"""
    state: str = Field(..., description="Robot state (STOPPED, MOVING, etc.)")
    linear_velocity: float = Field(..., description="Linear velocity in m/s")
    angular_velocity: float = Field(..., description="Angular velocity in rad/s")
    pedestrian_status: str = Field(default="none", description="Pedestrian detection status")
    lane_status: str = Field(default="none", description="Lane detection status")
    boundary_status: str = Field(default="none", description="Boundary detection status")
    fps: int = Field(default=0, description="Processing FPS")
    message: str = Field(default="", description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "state": "MOVING",
                "linear_velocity": 0.15,
                "angular_velocity": -0.2,
                "pedestrian_status": "none",
                "lane_status": "detected",
                "boundary_status": "clear",
                "fps": 28,
                "message": "Following lane"
            }
        }


class ManualControl(BaseModel):
    """Manual control command from web interface"""
    linear: float = Field(..., ge=-0.22, le=0.22, description="Linear velocity in m/s")
    angular: float = Field(..., ge=-2.84, le=2.84, description="Angular velocity in rad/s")

    class Config:
        json_schema_extra = {
            "example": {
                "linear": 0.15,
                "angular": 0.5
            }
        }


class ModeSwitch(BaseModel):
    """Mode switching command"""
    mode: Literal["manual", "autonomous"] = Field(..., description="Control mode")

    class Config:
        json_schema_extra = {
            "example": {
                "mode": "manual"
            }
        }


class CameraFrame(BaseModel):
    """Camera frame data"""
    frame: str = Field(..., description="Base64 encoded JPEG image")
    timestamp: float = Field(..., description="Timestamp of frame capture")

    class Config:
        json_schema_extra = {
            "example": {
                "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                "timestamp": 1638360600.123
            }
        }


class FleetSummary(BaseModel):
    """Fleet summary statistics"""
    total_vehicles: int
    active_count: int
    idle_count: int
    charging_count: int
    low_battery_count: int = Field(..., description="Vehicles with <20% battery")

    class Config:
        json_schema_extra = {
            "example": {
                "total_vehicles": 3,
                "active_count": 1,
                "idle_count": 1,
                "charging_count": 1,
                "low_battery_count": 0
            }
        }


class WebSocketMessage(BaseModel):
    """Generic WebSocket message structure"""
    type: str = Field(..., description="Message type (control, camera, status, fleet)")
    data: dict = Field(..., description="Message payload")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
