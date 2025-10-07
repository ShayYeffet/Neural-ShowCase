"""Configuration settings for the FastAPI backend."""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    app_name: str = "Neural Showcase API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # CORS Settings
    allowed_origins: List[str] = ["*"]
    allowed_methods: List[str] = ["*"]
    allowed_headers: List[str] = ["*"]
    
    # File Upload Settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_image_types: List[str] = ["image/jpeg", "image/png", "image/gif", "image/bmp"]
    allowed_text_types: List[str] = ["text/plain", "text/csv"]
    upload_dir: str = "uploads"
    
    # Model Settings
    model_registry_path: str = "models/registry"
    default_batch_size: int = 32
    max_batch_size: int = 100
    
    # Logging Settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security Settings
    secret_key: str = "your-secret-key-here"  # Change in production
    access_token_expire_minutes: int = 30
    
    # Performance Settings
    max_workers: int = 4
    timeout_seconds: int = 300
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()

# Ensure upload directory exists
upload_path = Path(settings.upload_dir)
upload_path.mkdir(exist_ok=True)