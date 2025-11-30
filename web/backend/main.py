"""
Vision360 Web Interface - FastAPI Backend
Main server application for TurtleBot autonomous driving web control
"""
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
from typing import List
import time

from models import (
    Vehicle, VehicleCreate, VehicleUpdate,
    ManualControl, ModeSwitch, FleetSummary
)
from mock_data import fleet_simulator, LOCATIONS, PRODUCT_TAGS
from websocket_manager import ws_manager
from ros_bridge import ROSBridgeThread


# Initialize FastAPI app
app = FastAPI(
    title="Vision360 Web Interface",
    description="Web control interface for TurtleBot autonomous driving system",
    version="1.0.0"
)

# Paths
BACKEND_DIR = Path(__file__).parent
WEB_DIR = BACKEND_DIR.parent
FRONTEND_DIR = WEB_DIR / "frontend"

# Mount static files (CSS, JS)
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

# Global instances
ros_bridge = ROSBridgeThread()
camera_queue = asyncio.Queue(maxsize=5)
status_queue = asyncio.Queue(maxsize=10)

# State
current_mode = "autonomous"  # manual or autonomous


@app.on_event("startup")
async def startup_event():
    """Initialize ROS bridge and background tasks on startup"""
    print("Starting Vision360 Web Interface...")

    # Start ROS bridge in separate thread
    ros_bridge.start(camera_queue, status_queue)

    # Start background tasks
    asyncio.create_task(broadcast_camera_loop())
    asyncio.create_task(broadcast_status_loop())
    asyncio.create_task(broadcast_fleet_loop())

    print("Web interface ready!")
    print(f"Frontend directory: {FRONTEND_DIR}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down...")
    ros_bridge.stop()


# ============================================================================
# Frontend Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve role selection page"""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/driver.html", response_class=HTMLResponse)
async def serve_driver():
    """Serve driver interface"""
    return FileResponse(FRONTEND_DIR / "driver.html")


@app.get("/manager.html", response_class=HTMLResponse)
async def serve_manager():
    """Serve manager interface"""
    return FileResponse(FRONTEND_DIR / "manager.html")


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time communication"""
    await ws_manager.connect(websocket)

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            message_type = data.get('type', '')

            # Handle different message types
            if message_type == 'control':
                # Manual control command
                await handle_manual_control(data.get('data', {}))

            elif message_type == 'ping':
                # Heartbeat
                await ws_manager.send_personal({'type': 'pong', 'data': {}}, websocket)

            elif message_type == 'request_frame':
                # Client requesting latest frame
                latest_frame = ros_bridge.get_latest_frame()
                if latest_frame:
                    await ws_manager.send_personal({
                        'type': 'camera',
                        'data': latest_frame
                    }, websocket)

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await ws_manager.disconnect(websocket)


async def handle_manual_control(data: dict):
    """Handle manual control command from web"""
    try:
        linear = float(data.get('linear', 0.0))
        angular = float(data.get('angular', 0.0))

        # Publish to ROS
        ros_bridge.publish_control(linear, angular)

    except Exception as e:
        print(f"Error handling manual control: {e}")


# ============================================================================
# Background Broadcasting Tasks
# ============================================================================

async def broadcast_camera_loop():
    """Broadcast camera frames to all connected clients"""
    while True:
        try:
            # Get frame from ROS bridge queue (with timeout)
            frame_data = await asyncio.wait_for(camera_queue.get(), timeout=0.1)
            await ws_manager.broadcast_camera(frame_data)
        except asyncio.TimeoutError:
            # No frame available, continue
            await asyncio.sleep(0.05)
        except Exception as e:
            print(f"Error in camera broadcast loop: {e}")
            await asyncio.sleep(0.1)


async def broadcast_status_loop():
    """Broadcast robot status to all connected clients"""
    while True:
        try:
            # Get status from ROS bridge queue (with timeout)
            status_data = await asyncio.wait_for(status_queue.get(), timeout=0.1)
            await ws_manager.broadcast_status(status_data)
        except asyncio.TimeoutError:
            # No status available, continue
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error in status broadcast loop: {e}")
            await asyncio.sleep(0.1)


async def broadcast_fleet_loop():
    """Broadcast fleet data to all connected clients (every 2 seconds)"""
    while True:
        try:
            # Get all vehicles
            vehicles = fleet_simulator.get_all_vehicles()
            summary = fleet_simulator.get_fleet_summary()

            fleet_data = {
                'vehicles': [v.model_dump(mode='json') for v in vehicles],
                'summary': summary.model_dump(mode='json')
            }

            await ws_manager.broadcast_fleet(fleet_data)
            await asyncio.sleep(2.0)  # Update every 2 seconds

        except Exception as e:
            print(f"Error in fleet broadcast loop: {e}")
            await asyncio.sleep(2.0)


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.post("/api/mode")
async def set_mode(mode_data: ModeSwitch):
    """Switch between manual and autonomous mode"""
    global current_mode
    current_mode = mode_data.mode

    # Publish to ROS
    ros_bridge.publish_mode(mode_data.mode)

    return {"status": "success", "mode": mode_data.mode}


@app.get("/api/mode")
async def get_mode():
    """Get current mode"""
    return {"mode": current_mode}


# ============================================================================
# Fleet Management API
# ============================================================================

@app.get("/api/vehicles", response_model=List[Vehicle])
async def get_vehicles():
    """Get all vehicles"""
    return fleet_simulator.get_all_vehicles()


@app.get("/api/vehicles/{vehicle_id}", response_model=Vehicle)
async def get_vehicle(vehicle_id: str):
    """Get specific vehicle"""
    vehicle = fleet_simulator.get_vehicle(vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return vehicle


@app.post("/api/vehicles", response_model=Vehicle)
async def create_vehicle(vehicle_data: VehicleCreate):
    """Create new vehicle (simulated)"""
    try:
        vehicle = fleet_simulator.add_vehicle(vehicle_data.model_dump())
        return vehicle
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/vehicles/{vehicle_id}", response_model=Vehicle)
async def update_vehicle(vehicle_id: str, vehicle_data: VehicleUpdate):
    """Update vehicle information"""
    # Filter out None values
    update_dict = {k: v for k, v in vehicle_data.model_dump().items() if v is not None}

    vehicle = fleet_simulator.update_vehicle(vehicle_id, update_dict)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")

    return vehicle


@app.delete("/api/vehicles/{vehicle_id}")
async def delete_vehicle(vehicle_id: str):
    """Delete vehicle (cannot delete real robot)"""
    success = fleet_simulator.delete_vehicle(vehicle_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete vehicle (either not found or is real robot)"
        )

    return {"status": "success", "message": f"Vehicle {vehicle_id} deleted"}


@app.get("/api/fleet/summary", response_model=FleetSummary)
async def get_fleet_summary():
    """Get fleet summary statistics"""
    return fleet_simulator.get_fleet_summary()


@app.get("/api/fleet/locations")
async def get_locations():
    """Get available locations for dropdowns"""
    return {"locations": LOCATIONS}


@app.get("/api/fleet/products")
async def get_products():
    """Get available product tags"""
    return {"products": PRODUCT_TAGS}


# ============================================================================
# Health Check
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": current_mode,
        "connections": ws_manager.get_connection_count(),
        "vehicles": len(fleet_simulator.get_all_vehicles())
    }


# ============================================================================
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ============================================================================
