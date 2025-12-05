"""
WebSocket Manager for Vision360
Handles WebSocket client connections and broadcasting
"""
from fastapi import WebSocket
from typing import Set, Dict
import json
import asyncio


class WebSocketManager:
    """Manages WebSocket connections and message broadcasting"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Accept and register new WebSocket connection"""
        await websocket.accept()
        async with self.lock:
            self.active_connections.add(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        async with self.lock:
            self.active_connections.discard(websocket)
        print(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending personal message: {e}")
            await self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        async with self.lock:
            connections = self.active_connections.copy()

        # Send to all clients, remove disconnected ones
        disconnected = set()
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        if disconnected:
            async with self.lock:
                self.active_connections -= disconnected

    async def broadcast_camera(self, frame_data: dict):
        """Broadcast camera frame to all clients"""
        message = {
            'type': 'camera',
            'data': frame_data
        }
        await self.broadcast(message)

    async def broadcast_status(self, status_data: dict):
        """Broadcast robot status to all clients"""
        message = {
            'type': 'status',
            'data': status_data
        }
        await self.broadcast(message)

    async def broadcast_fleet(self, fleet_data: dict):
        """Broadcast fleet data to all clients"""
        message = {
            'type': 'fleet',
            'data': fleet_data
        }
        await self.broadcast(message)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)


# Global WebSocket manager instance
ws_manager = WebSocketManager()
