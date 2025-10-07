"""Image classification endpoints with Grad-CAM visualization."""

import os
import sys
from pathlib import Path
import base64
import io
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import Optional, List
import logging
from datetime import datetime

from inference.api import NeuralShowcaseAPI
from ..utils import FileHandler, ResponseFormatter, validate_model_params
from ..config import settings
from ..main import PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/image", tags=["Image Classification"])

# Initialize neural API (will be injected)
neural_api = None

def set_neural_api(api: NeuralShowcaseAPI):
    """Set the neural API instance."""
    global neural_api
    neural_api = api

@router.post("/classify")
async def classify_image_endpoint(
    file: UploadFile = File(...),
    model_name: Optional[str] = Query(None, description="Name of the model to use"),
    model_id: Optional[str] = Query(None, description="Specific model ID to use"),
    return_probabilities: bool = Query(False, description="Return class probabilities")
):
    """
    Classify an uploaded image.
    
    - **file**: Image file to classify
    - **model_name**: Name of the CNN model to use (optional)
    - **model_id**: Specific model ID to use (optional)
    - **return_probabilities**: Whether to return class probabilities
    """
    if neural_api is None:
        raise HTTPException(status_code=500, detail="Neural API not initialized")
    
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

@router.post("/classify/gradcam")
async def classify_with_gradcam_endpoint(
    file: UploadFile = File(...),
    model_name: Optional[str] = Query(None, description="Name of the model to use"),
    model_id: Optional[str] = Query(None, description="Specific model ID to use")
):
    """
    Classify an image and return Grad-CAM visualization.
    
    - **file**: Image file to classify
    - **model_name**: Name of the CNN model to use (optional)
    - **model_id**: Specific model ID to use (optional)
    
    Returns classification result with Grad-CAM heatmap as base64 encoded image.
    """
    if neural_api is None:
        raise HTTPException(status_code=500, detail="Neural API not initialized")
    
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
        
        # Convert Grad-CAM heatmap to base64 image
        if 'gradcam_heatmap' in result:
            gradcam_b64 = _convert_gradcam_to_base64(result['gradcam_heatmap'])
            result['gradcam_image'] = gradcam_b64
        
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

@router.post("/classify/batch")
async def classify_batch_endpoint(
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = Query(None, description="Name of the model to use"),
    model_id: Optional[str] = Query(None, description="Specific model ID to use"),
    batch_size: int = Query(32, description="Batch size for processing")
):
    """
    Classify multiple images in batch.
    
    - **files**: List of image files to classify
    - **model_name**: Name of the CNN model to use (optional)
    - **model_id**: Specific model ID to use (optional)
    - **batch_size**: Batch size for processing
    """
    if neural_api is None:
        raise HTTPException(status_code=500, detail="Neural API not initialized")
    
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
    from ..utils import BatchProcessor
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

def _convert_gradcam_to_base64(gradcam_array: np.ndarray) -> str:
    """Convert Grad-CAM numpy array to base64 encoded image."""
    try:
        # Normalize the heatmap
        if gradcam_array.max() > gradcam_array.min():
            gradcam_normalized = (gradcam_array - gradcam_array.min()) / (gradcam_array.max() - gradcam_array.min())
        else:
            gradcam_normalized = gradcam_array
        
        # Create heatmap using matplotlib
        plt.figure(figsize=(8, 8))
        plt.imshow(gradcam_normalized, cmap='jet', alpha=0.8)
        plt.axis('off')
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Convert to base64
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{image_base64}"
    
    except Exception as e:
        logger.error(f"Error converting Grad-CAM to base64: {str(e)}")
        return None