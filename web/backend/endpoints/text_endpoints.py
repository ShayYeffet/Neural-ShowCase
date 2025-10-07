"""Text analysis endpoints with attention visualization."""

import os
import sys
from pathlib import Path
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

from inference.api import NeuralShowcaseAPI
from ..utils import FileHandler, ResponseFormatter, validate_model_params
from ..config import settings
from ..main import PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/text", tags=["Text Analysis"])

# Initialize neural API (will be injected)
neural_api = None

def set_neural_api(api: NeuralShowcaseAPI):
    """Set the neural API instance."""
    global neural_api
    neural_api = api

class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str
    model_name: Optional[str] = None
    model_id: Optional[str] = None
    return_attention: bool = False

@router.post("/analyze")
async def analyze_sentiment_endpoint(request: TextAnalysisRequest):
    """
    Analyze sentiment of text.
    
    - **text**: Text to analyze
    - **model_name**: Name of the transformer model to use (optional)
    - **model_id**: Specific model ID to use (optional)
    - **return_attention**: Whether to return attention weights
    """
    if neural_api is None:
        raise HTTPException(status_code=500, detail="Neural API not initialized")
    
    # Validate model parameters
    valid, error_msg = validate_model_params(request.model_name, request.model_id)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate text input
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(request.text) > 10000:  # Reasonable limit
        raise HTTPException(status_code=400, detail="Text too long (max 10000 characters)")
    
    try:
        # Analyze sentiment
        result = neural_api.analyze_sentiment(
            request.text,
            model_name=request.model_name,
            model_id=request.model_id,
            return_attention=request.return_attention
        )
        
        if 'error' in result:
            return PredictionResponse(
                success=False,
                error=result['error'],
                timestamp=datetime.now()
            )
        
        # Convert attention weights to visualization if present
        if 'attention_weights' in result and result['attention_weights'] is not None:
            attention_viz = _create_attention_visualization(
                request.text, 
                result['attention_weights']
            )
            if attention_viz:
                result['attention_visualization'] = attention_viz
        
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

@router.post("/analyze/file")
async def analyze_text_file_endpoint(
    file: UploadFile = File(...),
    model_name: Optional[str] = Query(None, description="Name of the model to use"),
    model_id: Optional[str] = Query(None, description="Specific model ID to use"),
    return_attention: bool = Query(False, description="Return attention weights")
):
    """
    Analyze sentiment of text from uploaded file.
    
    - **file**: Text file to analyze
    - **model_name**: Name of the transformer model to use (optional)
    - **model_id**: Specific model ID to use (optional)
    - **return_attention**: Whether to return attention weights
    """
    if neural_api is None:
        raise HTTPException(status_code=500, detail="Neural API not initialized")
    
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
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Analyze sentiment
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
        
        # Convert attention weights to visualization if present
        if 'attention_weights' in result and result['attention_weights'] is not None:
            attention_viz = _create_attention_visualization(text, result['attention_weights'])
            if attention_viz:
                result['attention_visualization'] = attention_viz
        
        # Add original text to response
        result['original_text'] = text[:500] + "..." if len(text) > 500 else text
        
        return PredictionResponse(
            success=True,
            data=result,
            timestamp=datetime.now()
        )
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8.")
    except Exception as e:
        logger.error(f"Error analyzing text file: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )

@router.post("/analyze/batch")
async def analyze_batch_texts_endpoint(
    texts: List[str] = Body(...),
    model_name: Optional[str] = Query(None, description="Name of the model to use"),
    model_id: Optional[str] = Query(None, description="Specific model ID to use"),
    batch_size: int = Query(16, description="Batch size for processing")
):
    """
    Analyze sentiment of multiple texts in batch.
    
    - **texts**: List of texts to analyze
    - **model_name**: Name of the transformer model to use (optional)
    - **model_id**: Specific model ID to use (optional)
    - **batch_size**: Batch size for processing
    """
    if neural_api is None:
        raise HTTPException(status_code=500, detail="Neural API not initialized")
    
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    # Validate model parameters
    valid, error_msg = validate_model_params(model_name, model_id)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate texts
    for i, text in enumerate(texts):
        if not text.strip():
            raise HTTPException(status_code=400, detail=f"Text at index {i} is empty")
        if len(text) > 10000:
            raise HTTPException(status_code=400, detail=f"Text at index {i} is too long")
    
    # Validate batch size
    from ..utils import BatchProcessor
    batch_size = BatchProcessor.validate_batch_size(batch_size)
    
    try:
        results = []
        
        # Process texts in batches
        text_batches = BatchProcessor.chunk_list(texts, batch_size)
        
        for batch in text_batches:
            batch_results = []
            for text in batch:
                result = neural_api.analyze_sentiment(
                    text,
                    model_name=model_name,
                    model_id=model_id,
                    return_attention=False  # Skip attention for batch processing
                )
                batch_results.append(result)
            results.extend(batch_results)
        
        return PredictionResponse(
            success=True,
            data={"results": results, "count": len(results)},
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error in batch text analysis: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )

def _create_attention_visualization(text: str, attention_weights: np.ndarray) -> Optional[str]:
    """Create attention visualization as base64 encoded image."""
    try:
        # Simple tokenization for visualization (in practice, use actual tokenizer)
        tokens = text.split()[:50]  # Limit to first 50 tokens for visualization
        
        # Ensure attention weights match token count
        if len(attention_weights.shape) > 1:
            # Take mean across attention heads if multi-head
            attention_weights = attention_weights.mean(axis=0)
        
        # Truncate or pad attention weights to match tokens
        if len(attention_weights) > len(tokens):
            attention_weights = attention_weights[:len(tokens)]
        elif len(attention_weights) < len(tokens):
            # Pad with zeros
            padding = np.zeros(len(tokens) - len(attention_weights))
            attention_weights = np.concatenate([attention_weights, padding])
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Create a matrix for visualization
        attention_matrix = attention_weights.reshape(1, -1)
        
        # Create heatmap
        sns.heatmap(
            attention_matrix,
            xticklabels=tokens,
            yticklabels=['Attention'],
            cmap='Blues',
            cbar=True,
            annot=False
        )
        
        plt.title('Attention Weights Visualization')
        plt.xlabel('Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # Convert to base64
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{image_base64}"
    
    except Exception as e:
        logger.error(f"Error creating attention visualization: {str(e)}")
        return None