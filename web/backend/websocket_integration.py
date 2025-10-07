"""Integration utilities for WebSocket notifications with inference API."""

import uuid
import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from .endpoints.websocket_endpoints import (
    notify_inference_started,
    notify_inference_progress, 
    notify_inference_complete,
    notify_training_started,
    notify_training_progress,
    notify_training_complete,
    notify_training_error,
    notify_model_update
)

logger = logging.getLogger(__name__)

class WebSocketInferenceWrapper:
    """Wrapper to add WebSocket notifications to inference operations."""
    
    def __init__(self, neural_api):
        self.neural_api = neural_api
    
    async def classify_image_with_notifications(self, image_path: str, model_name: str = None, 
                                              model_id: str = None, **kwargs):
        """Classify image with WebSocket notifications."""
        inference_id = str(uuid.uuid4())
        
        try:
            # Notify start
            await notify_inference_started(inference_id, "cnn", 1)
            
            # Perform inference
            result = self.neural_api.classify_image(
                image_path, model_name=model_name, model_id=model_id, **kwargs
            )
            
            # Notify completion
            await notify_inference_complete(inference_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in image classification: {str(e)}")
            await notify_inference_complete(inference_id, {"error": str(e)})
            raise
    
    async def classify_images_batch_with_notifications(self, image_paths: List[str], 
                                                     model_name: str = None, 
                                                     model_id: str = None, 
                                                     batch_size: int = 32):
        """Classify images in batch with WebSocket notifications."""
        inference_id = str(uuid.uuid4())
        
        try:
            # Notify start
            await notify_inference_started(inference_id, "cnn", len(image_paths))
            
            # Process in batches with progress updates
            results = []
            processed = 0
            
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                
                # Process batch
                batch_results = self.neural_api.classify_images_batch(
                    batch, model_name=model_name, model_id=model_id, batch_size=batch_size
                )
                
                results.extend(batch_results)
                processed += len(batch)
                
                # Notify progress
                await notify_inference_progress(inference_id, processed)
                
                # Small delay to allow other operations
                await asyncio.sleep(0.1)
            
            # Notify completion
            await notify_inference_complete(inference_id, {
                "results": results,
                "count": len(results)
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch image classification: {str(e)}")
            await notify_inference_complete(inference_id, {"error": str(e)})
            raise
    
    async def analyze_sentiment_with_notifications(self, text: str, model_name: str = None,
                                                 model_id: str = None, **kwargs):
        """Analyze sentiment with WebSocket notifications."""
        inference_id = str(uuid.uuid4())
        
        try:
            # Notify start
            await notify_inference_started(inference_id, "transformer", 1)
            
            # Perform inference
            result = self.neural_api.analyze_sentiment(
                text, model_name=model_name, model_id=model_id, **kwargs
            )
            
            # Notify completion
            await notify_inference_complete(inference_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            await notify_inference_complete(inference_id, {"error": str(e)})
            raise
    
    async def predict_timeseries_with_notifications(self, sequence: List[float], 
                                                  steps: int = 1, model_name: str = None,
                                                  model_id: str = None):
        """Predict time series with WebSocket notifications."""
        inference_id = str(uuid.uuid4())
        
        try:
            # Notify start
            await notify_inference_started(inference_id, "lstm", 1)
            
            # Perform inference
            result = self.neural_api.predict_timeseries(
                sequence, steps=steps, model_name=model_name, model_id=model_id
            )
            
            # Notify completion
            await notify_inference_complete(inference_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in time series prediction: {str(e)}")
            await notify_inference_complete(inference_id, {"error": str(e)})
            raise

class WebSocketTrainingWrapper:
    """Wrapper to add WebSocket notifications to training operations."""
    
    def __init__(self, training_manager=None):
        self.training_manager = training_manager
    
    async def start_training_with_notifications(self, model_type: str, config: Dict[str, Any]):
        """Start training with WebSocket notifications."""
        training_id = str(uuid.uuid4())
        
        try:
            # Notify start
            await notify_training_started(training_id, model_type, config)
            
            # This would integrate with the actual training system
            # For now, simulate training progress
            await self._simulate_training(training_id, config)
            
            return {"training_id": training_id, "status": "started"}
            
        except Exception as e:
            logger.error(f"Error starting training: {str(e)}")
            await notify_training_error(training_id, str(e))
            raise
    
    async def _simulate_training(self, training_id: str, config: Dict[str, Any]):
        """Simulate training progress for demonstration."""
        epochs = config.get("epochs", 10)
        
        for epoch in range(1, epochs + 1):
            # Simulate training time
            await asyncio.sleep(1)
            
            # Simulate decreasing loss
            loss = 1.0 * (0.9 ** epoch) + 0.1
            metrics = {
                "accuracy": min(0.95, 0.5 + (epoch / epochs) * 0.45),
                "val_loss": loss * 1.1,
                "val_accuracy": min(0.92, 0.45 + (epoch / epochs) * 0.47)
            }
            
            # Notify progress
            await notify_training_progress(training_id, epoch, loss, metrics)
        
        # Notify completion
        final_metrics = {
            "final_loss": loss,
            "final_accuracy": metrics["accuracy"],
            "final_val_loss": metrics["val_loss"],
            "final_val_accuracy": metrics["val_accuracy"],
            "total_epochs": epochs
        }
        
        await notify_training_complete(training_id, final_metrics)

# Utility functions for model management notifications

async def notify_model_registered(model_id: str, model_info: Dict[str, Any]):
    """Notify that a new model has been registered."""
    await notify_model_update(model_id, "registered", model_info)

async def notify_model_updated(model_id: str, update_info: Dict[str, Any]):
    """Notify that a model has been updated."""
    await notify_model_update(model_id, "updated", update_info)

async def notify_model_deleted(model_id: str):
    """Notify that a model has been deleted."""
    await notify_model_update(model_id, "deleted", {})

# Background task simulation functions (for testing)

async def simulate_training_session():
    """Simulate a training session for testing WebSocket functionality."""
    wrapper = WebSocketTrainingWrapper()
    
    config = {
        "model_type": "cnn",
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001
    }
    
    await wrapper.start_training_with_notifications("cnn", config)

async def simulate_batch_inference():
    """Simulate batch inference for testing WebSocket functionality."""
    # This would require actual image paths, so it's just a placeholder
    logger.info("Batch inference simulation would require actual image files")
    pass