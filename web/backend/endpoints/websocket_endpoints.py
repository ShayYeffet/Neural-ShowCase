"""WebSocket endpoints for real-time updates."""

import asyncio
import json
import uuid
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse

from ..websocket_manager import (
    connection_manager, 
    training_tracker, 
    inference_tracker,
    MessageType
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    subscriptions: Optional[str] = Query(None, description="Comma-separated list of subscriptions")
):
    """
    WebSocket endpoint for real-time updates.
    
    Subscriptions:
    - training: Training progress updates
    - inference: Inference progress updates  
    - models: Model updates
    - system: System status updates
    - all: All updates (default)
    """
    connection_id = str(uuid.uuid4())
    
    # Parse subscriptions
    subscription_list = []
    if subscriptions:
        subscription_list = [s.strip() for s in subscriptions.split(",")]
    
    # Connect to WebSocket
    connected = await connection_manager.connect(websocket, connection_id, subscription_list)
    
    if not connected:
        return
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_client_message(connection_id, message)
            except json.JSONDecodeError:
                await connection_manager.send_personal_message(connection_id, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Error handling client message: {str(e)}")
                await connection_manager.send_personal_message(connection_id, {
                    "type": "error", 
                    "message": "Error processing message"
                })
    
    except WebSocketDisconnect:
        connection_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {str(e)}")
        connection_manager.disconnect(connection_id)

async def handle_client_message(connection_id: str, message: dict):
    """Handle messages from WebSocket clients."""
    message_type = message.get("type")
    
    if message_type == "heartbeat":
        await connection_manager.handle_heartbeat(connection_id)
    
    elif message_type == "get_status":
        # Send current system status
        status = {
            "type": "system_status",
            "connections": connection_manager.get_connection_count(),
            "active_trainings": len(training_tracker.active_trainings),
            "active_inferences": len(inference_tracker.active_inferences),
            "timestamp": datetime.now().isoformat()
        }
        await connection_manager.send_personal_message(connection_id, status)
    
    elif message_type == "get_connection_info":
        # Send connection information (for admin/debug)
        info = connection_manager.get_connection_info()
        await connection_manager.send_personal_message(connection_id, {
            "type": "connection_info",
            "data": info
        })
    
    elif message_type == "subscribe":
        # Handle subscription changes
        new_subscriptions = message.get("subscriptions", [])
        # This would require updating the connection manager to support dynamic subscriptions
        await connection_manager.send_personal_message(connection_id, {
            "type": "subscription_updated",
            "subscriptions": new_subscriptions
        })
    
    else:
        await connection_manager.send_personal_message(connection_id, {
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        })

@router.get("/ws/test")
async def websocket_test_page():
    """Test page for WebSocket functionality."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neural Showcase WebSocket Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .message-box { 
                border: 1px solid #ccc; 
                height: 400px; 
                overflow-y: scroll; 
                padding: 10px; 
                margin: 10px 0;
                background-color: #f9f9f9;
            }
            .controls { margin: 10px 0; }
            button { margin: 5px; padding: 10px; }
            input, select { margin: 5px; padding: 5px; }
            .message { margin: 5px 0; padding: 5px; border-left: 3px solid #007cba; }
            .error { border-left-color: #d32f2f; }
            .success { border-left-color: #388e3c; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Neural Showcase WebSocket Test</h1>
            
            <div class="controls">
                <label>Subscriptions:</label>
                <select id="subscriptions" multiple>
                    <option value="all">All</option>
                    <option value="training">Training</option>
                    <option value="inference">Inference</option>
                    <option value="models">Models</option>
                    <option value="system">System</option>
                </select>
                <button onclick="connect()">Connect</button>
                <button onclick="disconnect()">Disconnect</button>
                <span id="status">Disconnected</span>
            </div>
            
            <div class="controls">
                <button onclick="sendHeartbeat()">Send Heartbeat</button>
                <button onclick="getStatus()">Get Status</button>
                <button onclick="getConnectionInfo()">Get Connection Info</button>
                <button onclick="simulateTraining()">Simulate Training</button>
                <button onclick="simulateInference()">Simulate Inference</button>
            </div>
            
            <div class="message-box" id="messages"></div>
            
            <div class="controls">
                <button onclick="clearMessages()">Clear Messages</button>
            </div>
        </div>

        <script>
            let ws = null;
            let messageCount = 0;

            function connect() {
                const subscriptions = Array.from(document.getElementById('subscriptions').selectedOptions)
                    .map(option => option.value).join(',');
                
                const wsUrl = `ws://localhost:8000/api/v1/ws?subscriptions=${subscriptions}`;
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function(event) {
                    document.getElementById('status').textContent = 'Connected';
                    addMessage('Connected to WebSocket', 'success');
                };
                
                ws.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    addMessage(`Received: ${JSON.stringify(message, null, 2)}`);
                };
                
                ws.onclose = function(event) {
                    document.getElementById('status').textContent = 'Disconnected';
                    addMessage('WebSocket connection closed', 'error');
                };
                
                ws.onerror = function(error) {
                    addMessage(`WebSocket error: ${error}`, 'error');
                };
            }

            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }

            function sendMessage(message) {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify(message));
                    addMessage(`Sent: ${JSON.stringify(message, null, 2)}`, 'success');
                } else {
                    addMessage('WebSocket not connected', 'error');
                }
            }

            function sendHeartbeat() {
                sendMessage({ type: 'heartbeat' });
            }

            function getStatus() {
                sendMessage({ type: 'get_status' });
            }

            function getConnectionInfo() {
                sendMessage({ type: 'get_connection_info' });
            }

            function simulateTraining() {
                // This would trigger a training simulation on the server
                addMessage('Training simulation not implemented in test page', 'error');
            }

            function simulateInference() {
                // This would trigger an inference simulation on the server
                addMessage('Inference simulation not implemented in test page', 'error');
            }

            function addMessage(text, type = '') {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}`;
                messageDiv.innerHTML = `<strong>[${++messageCount}]</strong> ${text.replace(/\\n/g, '<br>')}`;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function clearMessages() {
                document.getElementById('messages').innerHTML = '';
                messageCount = 0;
            }

            // Auto-connect on page load
            window.onload = function() {
                document.getElementById('subscriptions').selectedIndex = 0; // Select "All"
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Utility functions for triggering WebSocket events

async def notify_training_started(training_id: str, model_type: str, config: dict):
    """Notify clients that training has started."""
    await training_tracker.start_training(training_id, model_type, config)

async def notify_training_progress(training_id: str, epoch: int, loss: float, metrics: dict = None):
    """Notify clients of training progress."""
    await training_tracker.update_progress(training_id, epoch, loss, metrics)

async def notify_training_complete(training_id: str, final_metrics: dict):
    """Notify clients that training is complete."""
    await training_tracker.complete_training(training_id, final_metrics)

async def notify_training_error(training_id: str, error: str):
    """Notify clients of training error."""
    await training_tracker.error_training(training_id, error)

async def notify_inference_started(inference_id: str, model_type: str, batch_size: int = 1):
    """Notify clients that inference has started."""
    await inference_tracker.start_inference(inference_id, model_type, batch_size)

async def notify_inference_progress(inference_id: str, processed: int):
    """Notify clients of inference progress."""
    await inference_tracker.update_progress(inference_id, processed)

async def notify_inference_complete(inference_id: str, results: dict):
    """Notify clients that inference is complete."""
    await inference_tracker.complete_inference(inference_id, results)

async def notify_model_update(model_id: str, update_type: str, details: dict):
    """Notify clients of model updates."""
    await connection_manager.broadcast_to_subscription("models", {
        "type": MessageType.MODEL_UPDATE,
        "model_id": model_id,
        "update_type": update_type,
        "details": details
    })

async def notify_system_status(status: dict):
    """Notify clients of system status updates."""
    await connection_manager.broadcast_to_subscription("system", {
        "type": MessageType.SYSTEM_STATUS,
        "status": status
    })

# Background task for periodic status updates
async def periodic_status_updates():
    """Send periodic status updates to connected clients."""
    while True:
        try:
            if connection_manager.get_connection_count() > 0:
                status = {
                    "connections": connection_manager.get_connection_count(),
                    "active_trainings": len(training_tracker.active_trainings),
                    "active_inferences": len(inference_tracker.active_inferences),
                    "uptime": "N/A",  # Could track actual uptime
                    "memory_usage": "N/A"  # Could track actual memory usage
                }
                await notify_system_status(status)
            
            await asyncio.sleep(30)  # Update every 30 seconds
        except Exception as e:
            logger.error(f"Error in periodic status updates: {str(e)}")
            await asyncio.sleep(60)  # Wait longer on error