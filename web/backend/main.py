"""FastAPI backend for Neural Showcase web interface."""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import uvicorn
import asyncio
from datetime import datetime
import logging

# Simplified imports - removed problematic dependencies
import random
from .endpoints import image_endpoints, text_endpoints, timeseries_endpoints, websocket_endpoints

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="API for deep learning model demonstrations",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    debug=settings.debug
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

# Simplified - no complex neural API needed for demo

# Set neural API for endpoint modules
image_endpoints.set_neural_api(neural_api)
text_endpoints.set_neural_api(neural_api)
timeseries_endpoints.set_neural_api(neural_api)

# Include routers
app.include_router(image_endpoints.router, prefix="/api/v1")
app.include_router(text_endpoints.router, prefix="/api/v1")
app.include_router(timeseries_endpoints.router, prefix="/api/v1")
app.include_router(websocket_endpoints.router, prefix="/api/v1")

# Include monitoring endpoints
try:
    from .monitoring_endpoints import router as monitoring_router, cleanup_monitoring
    app.include_router(monitoring_router, prefix="/api/v1")
    logger.info("Monitoring endpoints included")
except ImportError as e:
    logger.warning(f"Failed to import monitoring endpoints: {e}")
    cleanup_monitoring = None

# Pydantic models for request/response
class PredictionResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime

class TextInput(BaseModel):
    text: str
    model_name: Optional[str] = None
    model_id: Optional[str] = None
    return_attention: bool = False

class TimeSeriesInput(BaseModel):
    sequence: List[float]
    steps: int = 1
    model_name: Optional[str] = None
    model_id: Optional[str] = None

class BatchImageInput(BaseModel):
    model_name: Optional[str] = None
    model_id: Optional[str] = None
    batch_size: int = 32

class ModelComparisonInput(BaseModel):
    model_ids: List[str]
    input_type: str  # 'image', 'text', or 'timeseries'

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

# Model management endpoints
@app.get("/models")
async def list_models(model_type: Optional[str] = None):
    """List available models."""
    try:
        models = neural_api.list_available_models(model_type)
        
        # If no models found, return some default demo models
        if not models:
            default_models = [
                {
                    'model_id': 'demo_cnn_resnet18',
                    'name': 'ResNet-18 (Demo)',
                    'type': 'cnn',
                    'version': '1.0.0',
                    'description': 'Pre-trained ResNet-18 for image classification',
                    'performance_metrics': {'accuracy': 0.85, 'loss': 0.45},
                    'created_at': '2024-01-01T00:00:00Z',
                    'tags': ['demo', 'pretrained']
                },
                {
                    'model_id': 'demo_transformer_bert',
                    'name': 'BERT-base (Demo)',
                    'type': 'transformer',
                    'version': '1.0.0',
                    'description': 'Pre-trained BERT for sentiment analysis',
                    'performance_metrics': {'accuracy': 0.88, 'f1_score': 0.87},
                    'created_at': '2024-01-01T00:00:00Z',
                    'tags': ['demo', 'pretrained']
                },
                {
                    'model_id': 'demo_lstm_timeseries',
                    'name': 'LSTM (Demo)',
                    'type': 'lstm',
                    'version': '1.0.0',
                    'description': 'LSTM model for time series forecasting',
                    'performance_metrics': {'mse': 0.12, 'mae': 0.08},
                    'created_at': '2024-01-01T00:00:00Z',
                    'tags': ['demo', 'timeseries']
                }
            ]
            
            # Filter by type if specified
            if model_type:
                default_models = [m for m in default_models if m['type'] == model_type]
            
            models = default_models
        
        return PredictionResponse(
            success=True,
            data={"models": models},
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )

@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get detailed information about a specific model."""
    try:
        info = neural_api.get_model_info(model_id)
        return PredictionResponse(
            success=True,
            data=info,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )

# Image classification endpoints
@app.post("/predict/image")
async def classify_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
    model_id: Optional[str] = None,
    return_probabilities: bool = False
):
    """Classify an uploaded image."""
    # Validate model parameters
    valid, error_msg = validate_model_params(model_name, model_id)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate image file
    valid, error_msg = FileHandler.validate_image_file(file)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    tmp_path = None
    try:
        # Save uploaded file temporarily
        tmp_path = FileHandler.save_temp_file(file, '.jpg')
        
        # Classify image
        result = neural_api.classify_image(
            tmp_path, 
            model_name=model_name,
            model_id=model_id,
            return_probabilities=return_probabilities
        )
        
        if 'error' in result:
            return PredictionResponse(
                success=False,
                error=result['error'],
                timestamp=datetime.now()
            )
        
        return PredictionResponse(
            success=True,
            data=result,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error classifying image: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )
    
    finally:
        # Clean up temporary file
        if tmp_path:
            FileHandler.cleanup_temp_file(tmp_path)

@app.post("/predict/image/gradcam")
async def classify_image_with_gradcam(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
    model_id: Optional[str] = None
):
    """Classify an image and return Grad-CAM visualization."""
    # Validate model parameters
    valid, error_msg = validate_model_params(model_name, model_id)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate image file
    valid, error_msg = FileHandler.validate_image_file(file)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    tmp_path = None
    try:
        # Save uploaded file temporarily
        tmp_path = FileHandler.save_temp_file(file, '.jpg')
        
        # Classify with Grad-CAM
        result = neural_api.classify_image_with_gradcam(
            tmp_path,
            model_name=model_name,
            model_id=model_id
        )
        
        if 'error' in result:
            return PredictionResponse(
                success=False,
                error=result['error'],
                timestamp=datetime.now()
            )
        
        return PredictionResponse(
            success=True,
            data=result,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error classifying image with Grad-CAM: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )
    
    finally:
        # Clean up temporary file
        if tmp_path:
            FileHandler.cleanup_temp_file(tmp_path)

@app.post("/predict/image/batch")
async def classify_images_batch(
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = None,
    model_id: Optional[str] = None,
    batch_size: int = 32
):
    """Classify multiple images in batch."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate model parameters
    valid, error_msg = validate_model_params(model_name, model_id)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate all files are images
    for file in files:
        valid, error_msg = FileHandler.validate_image_file(file)
        if not valid:
            raise HTTPException(status_code=400, detail=f"File {file.filename}: {error_msg}")
    
    # Validate batch size
    batch_size = BatchProcessor.validate_batch_size(batch_size)
    
    tmp_paths = []
    try:
        # Save all uploaded files temporarily
        tmp_paths = FileHandler.save_multiple_temp_files(files, '.jpg')
        
        # Classify images in batch
        results = neural_api.classify_images_batch(
            tmp_paths,
            model_name=model_name,
            model_id=model_id,
            batch_size=batch_size
        )
        
        return PredictionResponse(
            success=True,
            data={"results": results, "count": len(results)},
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error in batch image classification: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )
    
    finally:
        # Clean up temporary files
        FileHandler.cleanup_multiple_temp_files(tmp_paths)

# Text analysis endpoints
@app.post("/predict/text")
async def analyze_sentiment(input_data: TextInput):
    """Analyze sentiment of text."""
    try:
        result = neural_api.analyze_sentiment(
            input_data.text,
            model_name=input_data.model_name,
            model_id=input_data.model_id,
            return_attention=input_data.return_attention
        )
        
        if 'error' in result:
            return PredictionResponse(
                success=False,
                error=result['error'],
                timestamp=datetime.now()
            )
        
        return PredictionResponse(
            success=True,
            data=result,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )

@app.post("/predict/text/file")
async def analyze_text_file(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
    model_id: Optional[str] = None,
    return_attention: bool = False
):
    """Analyze sentiment of text from uploaded file."""
    # Validate model parameters
    valid, error_msg = validate_model_params(model_name, model_id)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate text file
    valid, error_msg = FileHandler.validate_text_file(file)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        # Read text from file
        content = await file.read()
        text = content.decode('utf-8')
        
        result = neural_api.analyze_sentiment(
            text,
            model_name=model_name,
            model_id=model_id,
            return_attention=return_attention
        )
        
        if 'error' in result:
            return PredictionResponse(
                success=False,
                error=result['error'],
                timestamp=datetime.now()
            )
        
        return PredictionResponse(
            success=True,
            data=result,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error analyzing text file: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )

# Time series prediction endpoints
@app.post("/predict/timeseries")
async def predict_timeseries(input_data: TimeSeriesInput):
    """Predict future values in time series."""
    try:
        result = neural_api.predict_timeseries(
            input_data.sequence,
            steps=input_data.steps,
            model_name=input_data.model_name,
            model_id=input_data.model_id
        )
        
        if 'error' in result:
            return PredictionResponse(
                success=False,
                error=result['error'],
                timestamp=datetime.now()
            )
        
        return PredictionResponse(
            success=True,
            data=result,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error predicting time series: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )

# Model comparison endpoint
@app.post("/compare/models")
async def compare_models(input_data: ModelComparisonInput):
    """Compare multiple models on the same input."""
    try:
        # This would need to be implemented based on input type
        # For now, return a placeholder
        return PredictionResponse(
            success=False,
            error="Model comparison not yet implemented",
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )

# Async prediction endpoints
@app.post("/predict/image/async")
async def classify_image_async(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
    model_id: Optional[str] = None
):
    """Asynchronous image classification."""
    # Validate model parameters
    valid, error_msg = validate_model_params(model_name, model_id)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate image file
    valid, error_msg = FileHandler.validate_image_file(file)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    tmp_path = None
    try:
        # Save uploaded file temporarily
        tmp_path = FileHandler.save_temp_file(file, '.jpg')
        
        # Classify image asynchronously (if async method exists)
        if hasattr(neural_api, 'classify_image_async'):
            result = await neural_api.classify_image_async(
                tmp_path,
                model_name=model_name,
                model_id=model_id
            )
        else:
            # Fallback to sync method
            result = neural_api.classify_image(
                tmp_path,
                model_name=model_name,
                model_id=model_id
            )
        
        if 'error' in result:
            return PredictionResponse(
                success=False,
                error=result['error'],
                timestamp=datetime.now()
            )
        
        return PredictionResponse(
            success=True,
            data=result,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error in async image classification: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )
    
    finally:
        # Clean up temporary file
        if tmp_path:
            FileHandler.cleanup_temp_file(tmp_path)

# Startup handler
@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on startup."""
    # Start periodic status updates
    asyncio.create_task(websocket_endpoints.periodic_status_updates())

# Shutdown handler
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    neural_api.shutdown()
    
    # Cleanup monitoring if available
    if cleanup_monitoring:
        cleanup_monitoring()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )