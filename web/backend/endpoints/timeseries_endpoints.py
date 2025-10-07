"""Time series prediction endpoints with forecasting visualization."""

import os
import sys
from pathlib import Path
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import logging
from datetime import datetime, timedelta

from inference.api import NeuralShowcaseAPI
from ..utils import FileHandler, ResponseFormatter, validate_model_params
from ..config import settings
from ..main import PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/timeseries", tags=["Time Series Prediction"])

# Initialize neural API (will be injected)
neural_api = None

def set_neural_api(api: NeuralShowcaseAPI):
    """Set the neural API instance."""
    global neural_api
    neural_api = api

class TimeSeriesPredictionRequest(BaseModel):
    """Request model for time series prediction."""
    sequence: List[float]
    steps: int = 1
    model_name: Optional[str] = None
    model_id: Optional[str] = None
    include_uncertainty: bool = True
    create_visualization: bool = True

class TimeSeriesFileRequest(BaseModel):
    """Request model for time series file upload."""
    steps: int = 1
    model_name: Optional[str] = None
    model_id: Optional[str] = None
    column_name: Optional[str] = None
    include_uncertainty: bool = True
    create_visualization: bool = True

@router.post("/predict")
async def predict_timeseries_endpoint(request: TimeSeriesPredictionRequest):
    """
    Predict future values in a time series.
    
    - **sequence**: Time series sequence data
    - **steps**: Number of steps to predict ahead
    - **model_name**: Name of the LSTM model to use (optional)
    - **model_id**: Specific model ID to use (optional)
    - **include_uncertainty**: Whether to include uncertainty estimates
    - **create_visualization**: Whether to create forecast visualization
    """
    if neural_api is None:
        raise HTTPException(status_code=500, detail="Neural API not initialized")
    
    # Validate model parameters
    valid, error_msg = validate_model_params(request.model_name, request.model_id)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate input sequence
    if not request.sequence:
        raise HTTPException(status_code=400, detail="Sequence cannot be empty")
    
    if len(request.sequence) < 10:
        raise HTTPException(status_code=400, detail="Sequence too short (minimum 10 points)")
    
    if len(request.sequence) > 10000:
        raise HTTPException(status_code=400, detail="Sequence too long (maximum 10000 points)")
    
    if request.steps < 1 or request.steps > 100:
        raise HTTPException(status_code=400, detail="Steps must be between 1 and 100")
    
    try:
        # Predict time series
        result = neural_api.predict_timeseries(
            request.sequence,
            steps=request.steps,
            model_name=request.model_name,
            model_id=request.model_id
        )
        
        if 'error' in result:
            return PredictionResponse(
                success=False,
                error=result['error'],
                timestamp=datetime.now()
            )
        
        # Create visualization if requested
        if request.create_visualization:
            viz = _create_forecast_visualization(
                request.sequence,
                result.get('predictions', []),
                result.get('uncertainty', None),
                request.steps
            )
            if viz:
                result['forecast_visualization'] = viz
        
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

@router.post("/predict/file")
async def predict_timeseries_file_endpoint(
    file: UploadFile = File(...),
    steps: int = Query(1, description="Number of steps to predict ahead"),
    model_name: Optional[str] = Query(None, description="Name of the model to use"),
    model_id: Optional[str] = Query(None, description="Specific model ID to use"),
    column_name: Optional[str] = Query(None, description="Column name for CSV files"),
    include_uncertainty: bool = Query(True, description="Include uncertainty estimates"),
    create_visualization: bool = Query(True, description="Create forecast visualization")
):
    """
    Predict time series from uploaded file (CSV or text).
    
    - **file**: CSV or text file containing time series data
    - **steps**: Number of steps to predict ahead
    - **model_name**: Name of the LSTM model to use (optional)
    - **model_id**: Specific model ID to use (optional)
    - **column_name**: Column name for CSV files (optional)
    - **include_uncertainty**: Whether to include uncertainty estimates
    - **create_visualization**: Whether to create forecast visualization
    """
    if neural_api is None:
        raise HTTPException(status_code=500, detail="Neural API not initialized")
    
    # Validate model parameters
    valid, error_msg = validate_model_params(model_name, model_id)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate file
    if file.content_type not in ['text/csv', 'text/plain', 'application/csv']:
        raise HTTPException(status_code=400, detail="File must be CSV or text format")
    
    if steps < 1 or steps > 100:
        raise HTTPException(status_code=400, detail="Steps must be between 1 and 100")
    
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Parse time series data
        sequence = _parse_timeseries_file(text_content, column_name)
        
        if not sequence:
            raise HTTPException(status_code=400, detail="Could not parse time series data from file")
        
        if len(sequence) < 10:
            raise HTTPException(status_code=400, detail="Time series too short (minimum 10 points)")
        
        # Predict time series
        result = neural_api.predict_timeseries(
            sequence,
            steps=steps,
            model_name=model_name,
            model_id=model_id
        )
        
        if 'error' in result:
            return PredictionResponse(
                success=False,
                error=result['error'],
                timestamp=datetime.now()
            )
        
        # Create visualization if requested
        if create_visualization:
            viz = _create_forecast_visualization(
                sequence,
                result.get('predictions', []),
                result.get('uncertainty', None),
                steps
            )
            if viz:
                result['forecast_visualization'] = viz
        
        # Add original data info
        result['original_data_points'] = len(sequence)
        result['original_data_sample'] = sequence[-20:]  # Last 20 points
        
        return PredictionResponse(
            success=True,
            data=result,
            timestamp=datetime.now()
        )
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8.")
    except Exception as e:
        logger.error(f"Error predicting time series from file: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )

@router.post("/forecast/multi")
async def multi_step_forecast_endpoint(
    sequence: List[float] = Body(...),
    forecast_horizons: List[int] = Body([1, 5, 10], description="Multiple forecast horizons"),
    model_name: Optional[str] = Query(None, description="Name of the model to use"),
    model_id: Optional[str] = Query(None, description="Specific model ID to use"),
    create_visualization: bool = Query(True, description="Create forecast visualization")
):
    """
    Create multi-horizon forecasts for a time series.
    
    - **sequence**: Time series sequence data
    - **forecast_horizons**: List of forecast horizons to compute
    - **model_name**: Name of the LSTM model to use (optional)
    - **model_id**: Specific model ID to use (optional)
    - **create_visualization**: Whether to create forecast visualization
    """
    if neural_api is None:
        raise HTTPException(status_code=500, detail="Neural API not initialized")
    
    # Validate model parameters
    valid, error_msg = validate_model_params(model_name, model_id)
    if not valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate inputs
    if not sequence:
        raise HTTPException(status_code=400, detail="Sequence cannot be empty")
    
    if len(sequence) < 10:
        raise HTTPException(status_code=400, detail="Sequence too short (minimum 10 points)")
    
    if not forecast_horizons or max(forecast_horizons) > 100:
        raise HTTPException(status_code=400, detail="Invalid forecast horizons")
    
    try:
        results = {}
        
        # Generate forecasts for each horizon
        for horizon in forecast_horizons:
            result = neural_api.predict_timeseries(
                sequence,
                steps=horizon,
                model_name=model_name,
                model_id=model_id
            )
            
            if 'error' not in result:
                results[f"horizon_{horizon}"] = result
        
        if not results:
            return PredictionResponse(
                success=False,
                error="Failed to generate any forecasts",
                timestamp=datetime.now()
            )
        
        # Create multi-horizon visualization
        if create_visualization:
            viz = _create_multi_horizon_visualization(sequence, results)
            if viz:
                results['multi_horizon_visualization'] = viz
        
        return PredictionResponse(
            success=True,
            data=results,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error in multi-step forecast: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )

def _parse_timeseries_file(content: str, column_name: Optional[str] = None) -> List[float]:
    """Parse time series data from file content."""
    try:
        lines = content.strip().split('\n')
        
        # Try to parse as CSV first
        if ',' in content:
            import csv
            from io import StringIO
            
            csv_reader = csv.reader(StringIO(content))
            rows = list(csv_reader)
            
            if len(rows) > 1:  # Has header
                header = rows[0]
                data_rows = rows[1:]
                
                # Find column index
                col_idx = 0
                if column_name:
                    try:
                        col_idx = header.index(column_name)
                    except ValueError:
                        # Column not found, try first numeric column
                        for i, col in enumerate(header):
                            try:
                                float(data_rows[0][i])
                                col_idx = i
                                break
                            except (ValueError, IndexError):
                                continue
                
                # Extract values
                values = []
                for row in data_rows:
                    try:
                        values.append(float(row[col_idx]))
                    except (ValueError, IndexError):
                        continue
                
                return values
        
        # Try to parse as simple numeric values
        values = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    # Try to extract first number from line
                    parts = line.replace(',', ' ').split()
                    for part in parts:
                        try:
                            values.append(float(part))
                            break
                        except ValueError:
                            continue
                except:
                    continue
        
        return values
    
    except Exception as e:
        logger.error(f"Error parsing time series file: {str(e)}")
        return []

def _create_forecast_visualization(
    historical: List[float], 
    predictions: Union[List[float], np.ndarray], 
    uncertainty: Optional[Union[List[float], np.ndarray]] = None,
    steps: int = 1
) -> Optional[str]:
    """Create forecast visualization as base64 encoded image."""
    try:
        # Convert to numpy arrays
        historical = np.array(historical)
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        
        # Create time indices
        hist_time = np.arange(len(historical))
        pred_time = np.arange(len(historical), len(historical) + len(predictions))
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot historical data
        plt.plot(hist_time, historical, 'b-', label='Historical Data', linewidth=2)
        
        # Plot predictions
        plt.plot(pred_time, predictions, 'r-', label='Predictions', linewidth=2, marker='o')
        
        # Plot uncertainty bands if available
        if uncertainty is not None:
            uncertainty = np.array(uncertainty)
            plt.fill_between(
                pred_time,
                predictions - uncertainty,
                predictions + uncertainty,
                alpha=0.3,
                color='red',
                label='Uncertainty'
            )
        
        # Add vertical line at prediction start
        plt.axvline(x=len(historical)-1, color='gray', linestyle='--', alpha=0.7)
        
        plt.title(f'Time Series Forecast ({steps} steps ahead)')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
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
        logger.error(f"Error creating forecast visualization: {str(e)}")
        return None

def _create_multi_horizon_visualization(
    historical: List[float], 
    results: Dict[str, Any]
) -> Optional[str]:
    """Create multi-horizon forecast visualization."""
    try:
        historical = np.array(historical)
        hist_time = np.arange(len(historical))
        
        plt.figure(figsize=(14, 10))
        
        # Plot historical data
        plt.plot(hist_time, historical, 'b-', label='Historical Data', linewidth=2)
        
        # Plot forecasts for different horizons
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for i, (horizon_key, result) in enumerate(results.items()):
            if horizon_key.startswith('horizon_'):
                horizon = int(horizon_key.split('_')[1])
                predictions = np.array(result.get('predictions', []))
                
                if len(predictions) > 0:
                    pred_time = np.arange(len(historical), len(historical) + len(predictions))
                    color = colors[i % len(colors)]
                    
                    plt.plot(
                        pred_time, 
                        predictions, 
                        color=color, 
                        label=f'{horizon}-step forecast',
                        linewidth=2,
                        marker='o',
                        markersize=4
                    )
        
        # Add vertical line at prediction start
        plt.axvline(x=len(historical)-1, color='gray', linestyle='--', alpha=0.7)
        
        plt.title('Multi-Horizon Time Series Forecasts')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
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
        logger.error(f"Error creating multi-horizon visualization: {str(e)}")
        return None