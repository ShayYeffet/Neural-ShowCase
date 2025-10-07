"""WebSocket manager for real-time updates."""

import asyncio
import json
import logging
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from enum import Enum

logger = logging.getLogger(__name__)

class MessageType(str, Enum):
    """WebSocket message types."""
    TRAINING_PROGRESS = "training_progress"
    TRAINING_COMPLETE = "training_complete"
    TRAINING_ERROR = "training_error"
    INFERENCE_START = "inference_start"
    INFERENCE_PROGRESS = "inference_progress"
    INFERENCE_COMPLETE = "inference_complete"
    INFERENCE_ERROR = "inference_error"
    MODEL_UPDATE = "model_update"
    SYSTEM_STATUS = "system_status"
    HEARTBEAT = "heartbeat"

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        # Store active connections by connection ID
        self.active_connections: Dict[str, WebSocket] = {}
        # Store connections by subscription type
        self.subscriptions: Dict[str, Set[str]] = {
            "training": set(),
            "inference": set(),
            "models": set(),
            "system": set(),
            "all": set()
        }
        # Store connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, connection_id: str, 
                     subscriptions: List[str] = None) -> bool:
        """Accept a WebSocket connection and register subscriptions."""
        try:
            await websocket.accept()
            self.active_connections[connection_id] = websocket
            
            # Set up subscriptions
            if subscriptions:
                for sub_type in subscriptions:
                    if sub_type in self.subscriptions:
                        self.subscriptions[sub_type].add(connection_id)
            else:
                # Default to all subscriptions
                self.subscriptions["all"].add(connection_id)
            
            # Store metadata
            self.connection_metadata[connection_id] = {
                "connected_at": datetime.now(),
                "subscriptions": subscriptions or ["all"],
                "last_heartbeat": datetime.now()
            }
            
            logger.info(f"WebSocket connection {connection_id} established")
            
            # Send welcome message
            await self.send_personal_message(connection_id, {
                "type": "connection_established",
                "connection_id": connection_id,
                "subscriptions": subscriptions or ["all"],
                "timestamp": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket {connection_id}: {str(e)}")
            return False
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
        # Remove from active connections
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        # Remove from all subscriptions
        for sub_set in self.subscriptions.values():
            sub_set.discard(connection_id)
        
        # Remove metadata
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]
        
        logger.info(f"WebSocket connection {connection_id} disconnected")
    
    async def send_personal_message(self, connection_id: str, message: Dict[str, Any]):
        """Send a message to a specific connection."""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {str(e)}")
                # Remove broken connection
                self.disconnect(connection_id)
    
    async def broadcast_to_subscription(self, subscription_type: str, message: Dict[str, Any]):
        """Broadcast a message to all connections with a specific subscription."""
        if subscription_type not in self.subscriptions:
            logger.warning(f"Unknown subscription type: {subscription_type}")
            return
        
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()
        
        # Send to specific subscription
        connection_ids = list(self.subscriptions[subscription_type])
        for connection_id in connection_ids:
            await self.send_personal_message(connection_id, message)
        
        # Also send to "all" subscribers
        all_connection_ids = list(self.subscriptions["all"])
        for connection_id in all_connection_ids:
            if connection_id not in connection_ids:  # Avoid duplicates
                await self.send_personal_message(connection_id, message)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast a message to all active connections."""
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()
        
        connection_ids = list(self.active_connections.keys())
        for connection_id in connection_ids:
            await self.send_personal_message(connection_id, message)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
    
    def get_subscription_count(self, subscription_type: str) -> int:
        """Get the number of connections for a subscription type."""
        return len(self.subscriptions.get(subscription_type, set()))
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about all connections."""
        return {
            "total_connections": len(self.active_connections),
            "subscriptions": {
                sub_type: len(connections) 
                for sub_type, connections in self.subscriptions.items()
            },
            "connections": {
                conn_id: {
                    "connected_at": metadata["connected_at"].isoformat(),
                    "subscriptions": metadata["subscriptions"],
                    "last_heartbeat": metadata["last_heartbeat"].isoformat()
                }
                for conn_id, metadata in self.connection_metadata.items()
            }
        }
    
    async def handle_heartbeat(self, connection_id: str):
        """Handle heartbeat from a connection."""
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["last_heartbeat"] = datetime.now()
            
            await self.send_personal_message(connection_id, {
                "type": MessageType.HEARTBEAT,
                "status": "alive",
                "timestamp": datetime.now().isoformat()
            })

# Global connection manager instance
connection_manager = ConnectionManager()

class TrainingProgressTracker:
    """Track and broadcast training progress."""
    
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.active_trainings: Dict[str, Dict[str, Any]] = {}
    
    async def start_training(self, training_id: str, model_type: str, 
                           config: Dict[str, Any]):
        """Start tracking a training session."""
        self.active_trainings[training_id] = {
            "model_type": model_type,
            "config": config,
            "started_at": datetime.now(),
            "status": "started",
            "current_epoch": 0,
            "total_epochs": config.get("epochs", 0),
            "current_loss": None,
            "best_loss": None,
            "metrics": {}
        }
        
        await self.manager.broadcast_to_subscription("training", {
            "type": MessageType.TRAINING_PROGRESS,
            "training_id": training_id,
            "status": "started",
            "model_type": model_type,
            "config": config
        })
    
    async def update_progress(self, training_id: str, epoch: int, 
                            loss: float, metrics: Dict[str, float] = None):
        """Update training progress."""
        if training_id not in self.active_trainings:
            return
        
        training_info = self.active_trainings[training_id]
        training_info["current_epoch"] = epoch
        training_info["current_loss"] = loss
        training_info["metrics"] = metrics or {}
        
        if training_info["best_loss"] is None or loss < training_info["best_loss"]:
            training_info["best_loss"] = loss
        
        await self.manager.broadcast_to_subscription("training", {
            "type": MessageType.TRAINING_PROGRESS,
            "training_id": training_id,
            "status": "training",
            "epoch": epoch,
            "total_epochs": training_info["total_epochs"],
            "loss": loss,
            "best_loss": training_info["best_loss"],
            "metrics": metrics or {},
            "progress_percent": (epoch / training_info["total_epochs"]) * 100 if training_info["total_epochs"] > 0 else 0
        })
    
    async def complete_training(self, training_id: str, final_metrics: Dict[str, Any]):
        """Mark training as complete."""
        if training_id not in self.active_trainings:
            return
        
        training_info = self.active_trainings[training_id]
        training_info["status"] = "completed"
        training_info["completed_at"] = datetime.now()
        training_info["final_metrics"] = final_metrics
        
        await self.manager.broadcast_to_subscription("training", {
            "type": MessageType.TRAINING_COMPLETE,
            "training_id": training_id,
            "status": "completed",
            "final_metrics": final_metrics,
            "duration": (datetime.now() - training_info["started_at"]).total_seconds()
        })
        
        # Remove from active trainings after a delay
        await asyncio.sleep(60)  # Keep for 1 minute
        if training_id in self.active_trainings:
            del self.active_trainings[training_id]
    
    async def error_training(self, training_id: str, error: str):
        """Mark training as failed."""
        if training_id not in self.active_trainings:
            return
        
        training_info = self.active_trainings[training_id]
        training_info["status"] = "error"
        training_info["error"] = error
        training_info["failed_at"] = datetime.now()
        
        await self.manager.broadcast_to_subscription("training", {
            "type": MessageType.TRAINING_ERROR,
            "training_id": training_id,
            "status": "error",
            "error": error
        })
        
        # Remove from active trainings
        del self.active_trainings[training_id]

class InferenceProgressTracker:
    """Track and broadcast inference progress."""
    
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.active_inferences: Dict[str, Dict[str, Any]] = {}
    
    async def start_inference(self, inference_id: str, model_type: str, 
                            batch_size: int = 1):
        """Start tracking an inference session."""
        self.active_inferences[inference_id] = {
            "model_type": model_type,
            "batch_size": batch_size,
            "started_at": datetime.now(),
            "status": "started",
            "processed": 0,
            "total": batch_size
        }
        
        await self.manager.broadcast_to_subscription("inference", {
            "type": MessageType.INFERENCE_START,
            "inference_id": inference_id,
            "status": "started",
            "model_type": model_type,
            "batch_size": batch_size
        })
    
    async def update_progress(self, inference_id: str, processed: int):
        """Update inference progress."""
        if inference_id not in self.active_inferences:
            return
        
        inference_info = self.active_inferences[inference_id]
        inference_info["processed"] = processed
        
        await self.manager.broadcast_to_subscription("inference", {
            "type": MessageType.INFERENCE_PROGRESS,
            "inference_id": inference_id,
            "status": "processing",
            "processed": processed,
            "total": inference_info["total"],
            "progress_percent": (processed / inference_info["total"]) * 100
        })
    
    async def complete_inference(self, inference_id: str, results: Dict[str, Any]):
        """Mark inference as complete."""
        if inference_id not in self.active_inferences:
            return
        
        inference_info = self.active_inferences[inference_id]
        inference_info["status"] = "completed"
        inference_info["completed_at"] = datetime.now()
        
        await self.manager.broadcast_to_subscription("inference", {
            "type": MessageType.INFERENCE_COMPLETE,
            "inference_id": inference_id,
            "status": "completed",
            "results": results,
            "duration": (datetime.now() - inference_info["started_at"]).total_seconds()
        })
        
        # Remove from active inferences
        del self.active_inferences[inference_id]

# Global trackers
training_tracker = TrainingProgressTracker(connection_manager)
inference_tracker = InferenceProgressTracker(connection_manager)