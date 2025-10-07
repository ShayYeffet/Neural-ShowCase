"""Utility functions for the FastAPI backend."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import magic
from fastapi import UploadFile, HTTPException
import logging

from web.backend.config import settings

logger = logging.getLogger(__name__)

class FileHandler:
    """Handle file operations for the API."""
    
    @staticmethod
    def validate_file_type(file: UploadFile, allowed_types: List[str]) -> bool:
        """Validate if file type is allowed."""
        return file.content_type in allowed_types
    
    @staticmethod
    def validate_file_size(file: UploadFile, max_size: int = None) -> bool:
        """Validate file size."""
        if max_size is None:
            max_size = settings.max_file_size
        
        # Get file size
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        return size <= max_size
    
    @staticmethod
    def validate_image_file(file: UploadFile) -> Tuple[bool, Optional[str]]:
        """Validate image file."""
        if not FileHandler.validate_file_type(file, settings.allowed_image_types):
            return False, f"Invalid file type. Allowed types: {settings.allowed_image_types}"
        
        if not FileHandler.validate_file_size(file):
            return False, f"File too large. Maximum size: {settings.max_file_size} bytes"
        
        return True, None
    
    @staticmethod
    def validate_text_file(file: UploadFile) -> Tuple[bool, Optional[str]]:
        """Validate text file."""
        if not FileHandler.validate_file_type(file, settings.allowed_text_types):
            return False, f"Invalid file type. Allowed types: {settings.allowed_text_types}"
        
        if not FileHandler.validate_file_size(file):
            return False, f"File too large. Maximum size: {settings.max_file_size} bytes"
        
        return True, None
    
    @staticmethod
    def save_temp_file(file: UploadFile, suffix: str = None) -> str:
        """Save uploaded file to temporary location."""
        if suffix is None:
            suffix = Path(file.filename).suffix if file.filename else '.tmp'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            return tmp_file.name
    
    @staticmethod
    def cleanup_temp_file(file_path: str):
        """Clean up temporary file."""
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    @staticmethod
    def save_multiple_temp_files(files: List[UploadFile], suffix: str = None) -> List[str]:
        """Save multiple uploaded files to temporary locations."""
        temp_paths = []
        try:
            for file in files:
                temp_path = FileHandler.save_temp_file(file, suffix)
                temp_paths.append(temp_path)
            return temp_paths
        except Exception as e:
            # Cleanup any files that were saved
            for path in temp_paths:
                FileHandler.cleanup_temp_file(path)
            raise e
    
    @staticmethod
    def cleanup_multiple_temp_files(file_paths: List[str]):
        """Clean up multiple temporary files."""
        for path in file_paths:
            FileHandler.cleanup_temp_file(path)

class ResponseFormatter:
    """Format API responses consistently."""
    
    @staticmethod
    def success_response(data: dict, message: str = None):
        """Format successful response."""
        response = {
            "success": True,
            "data": data
        }
        if message:
            response["message"] = message
        return response
    
    @staticmethod
    def error_response(error: str, code: int = None):
        """Format error response."""
        response = {
            "success": False,
            "error": error
        }
        if code:
            response["error_code"] = code
        return response
    
    @staticmethod
    def validation_error_response(errors: List[str]):
        """Format validation error response."""
        return {
            "success": False,
            "error": "Validation failed",
            "validation_errors": errors
        }

class BatchProcessor:
    """Handle batch processing operations."""
    
    @staticmethod
    def validate_batch_size(batch_size: int) -> int:
        """Validate and adjust batch size."""
        if batch_size <= 0:
            return settings.default_batch_size
        if batch_size > settings.max_batch_size:
            return settings.max_batch_size
        return batch_size
    
    @staticmethod
    def chunk_list(items: List, chunk_size: int) -> List[List]:
        """Split list into chunks."""
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def validate_model_params(model_name: Optional[str], model_id: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Validate model parameters."""
    if model_name and model_id:
        return False, "Cannot specify both model_name and model_id"
    return True, None

def handle_api_error(func):
    """Decorator to handle API errors consistently."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"API error in {func.__name__}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    return wrapper