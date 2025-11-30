"""
Mock data generator for fleet management
Creates demo vehicles with South Korean locations for customer presentation
"""
import random
from typing import Dict, List
from models import Vehicle, FleetSummary


# South Korean locations for journey simulation
LOCATIONS = [
    "Incheon Airport Warehouse",
    "Incheon Port Logistics Center",
    "Gangnam Distribution Hub",
    "Hongdae Fulfillment Center",
    "Songdo Smart City Depot",
    "Busan Port Terminal",
    "Gwangjin Loading Dock",
    "Yongsan Distribution Center",
    "Itaewon Customer Pickup",
    "Myeongdong Retail Center",
    "Dongdaemun Fashion Hub",
    "Jamsil Sports Complex Warehouse",
    "Yeouido Business District Depot",
    "Gangbuk Medical Supply Center",
    "Suwon Industrial Park"
]

PRODUCT_TAGS = [
    "Electronics",
    "Medical Supplies",
    "Food & Beverages",
    "Industrial Parts",
    "Retail Goods",
    "Pharmaceuticals",
    "Office Supplies",
    "Fashion Items",
    "Automotive Parts"
]

# Korean addresses for display (fake but realistic-looking)
ADDRESSES = [
    "37.4563° N, 126.7052° E - Incheon",
    "37.5665° N, 126.9780° E - Seoul",
    "37.3886° N, 126.6432° E - Songdo",
    "37.5172° N, 127.0473° E - Gangnam",
    "35.1796° N, 129.0756° E - Busan",
    "37.5326° N, 127.0246° E - Gwangjin",
    "37.5219° N, 126.9245° E - Yongsan",
    "37.5547° N, 126.9707° E - Hongdae"
]


class FleetSimulator:
    """Manages mock fleet data for demonstration"""

    def __init__(self):
        self.vehicles: Dict[str, Vehicle] = {}
        self._initialize_fleet()

    def _initialize_fleet(self):
        """Initialize fleet with real robot + demo simulated vehicles"""
        # Real robot (TB3-001)
        self.vehicles["TB3-001"] = Vehicle(
            id="TB3-001",
            model="TurtleBot3 Burger",
            product_tag="Electronics",
            departure="Incheon Airport Warehouse",
            destination="Gangnam Distribution Hub",
            latitude=37.4563,
            longitude=126.7052,
            battery_percent=87,
            status="active",
            is_real=True
        )

        # Demo simulated vehicle 1 (TB3-002)
        self.vehicles["TB3-002"] = Vehicle(
            id="TB3-002",
            model="TurtleBot3 Burger",
            product_tag="Medical Supplies",
            departure="Gangbuk Medical Supply Center",
            destination="Songdo Smart City Depot",
            latitude=37.5665,
            longitude=126.9780,
            battery_percent=45,
            status="charging",
            is_real=False
        )

        # Demo simulated vehicle 2 (TB3-003)
        self.vehicles["TB3-003"] = Vehicle(
            id="TB3-003",
            model="TurtleBot3 Burger",
            product_tag="Fashion Items",
            departure="Dongdaemun Fashion Hub",
            destination="Myeongdong Retail Center",
            latitude=37.5172,
            longitude=127.0473,
            battery_percent=92,
            status="idle",
            is_real=False
        )

    def get_all_vehicles(self) -> List[Vehicle]:
        """Get all vehicles as list"""
        return list(self.vehicles.values())

    def get_vehicle(self, vehicle_id: str) -> Vehicle | None:
        """Get specific vehicle by ID"""
        return self.vehicles.get(vehicle_id)

    def add_vehicle(self, vehicle_data: dict) -> Vehicle:
        """Add new demo vehicle"""
        # Set random Korean location if not provided
        if "latitude" not in vehicle_data or vehicle_data["latitude"] is None:
            random_addr = random.choice(ADDRESSES)
            lat_str = random_addr.split("°")[0]
            vehicle_data["latitude"] = float(lat_str)

        if "longitude" not in vehicle_data or vehicle_data["longitude"] is None:
            random_addr = random.choice(ADDRESSES)
            lon_str = random_addr.split(",")[1].split("°")[0].strip()
            vehicle_data["longitude"] = float(lon_str)

        # Set random locations if not provided
        if "departure" not in vehicle_data:
            vehicle_data["departure"] = random.choice(LOCATIONS)
        if "destination" not in vehicle_data:
            vehicle_data["destination"] = random.choice(LOCATIONS)

        vehicle = Vehicle(**vehicle_data, is_real=False)
        self.vehicles[vehicle.id] = vehicle
        return vehicle

    def update_vehicle(self, vehicle_id: str, update_data: dict) -> Vehicle | None:
        """Update vehicle information"""
        if vehicle_id not in self.vehicles:
            return None

        vehicle = self.vehicles[vehicle_id]

        # Don't allow editing certain fields of real robot
        if vehicle.is_real:
            # Only allow status updates for real robot
            allowed_fields = ["status"]
            update_data = {k: v for k, v in update_data.items() if k in allowed_fields}

        # Update vehicle fields
        for key, value in update_data.items():
            if hasattr(vehicle, key) and value is not None:
                setattr(vehicle, key, value)

        return vehicle

    def delete_vehicle(self, vehicle_id: str) -> bool:
        """Delete vehicle (cannot delete real robot)"""
        if vehicle_id not in self.vehicles:
            return False

        if self.vehicles[vehicle_id].is_real:
            return False  # Cannot delete real robot

        del self.vehicles[vehicle_id]
        return True

    def update_real_robot_battery(self, battery_percent: int):
        """Update real robot's battery (for future real sensor integration)"""
        if "TB3-001" in self.vehicles:
            self.vehicles["TB3-001"].battery_percent = max(0, min(100, battery_percent))

    def update_real_robot_location(self, latitude: float, longitude: float):
        """Update real robot's location (for future GPS/odometry integration)"""
        if "TB3-001" in self.vehicles:
            self.vehicles["TB3-001"].latitude = latitude
            self.vehicles["TB3-001"].longitude = longitude

    def get_fleet_summary(self) -> FleetSummary:
        """Get fleet statistics"""
        vehicles = list(self.vehicles.values())
        return FleetSummary(
            total_vehicles=len(vehicles),
            active_count=sum(1 for v in vehicles if v.status == "active"),
            idle_count=sum(1 for v in vehicles if v.status == "idle"),
            charging_count=sum(1 for v in vehicles if v.status == "charging"),
            low_battery_count=sum(1 for v in vehicles if v.battery_percent < 20)
        )


# Global fleet simulator instance
fleet_simulator = FleetSimulator()
