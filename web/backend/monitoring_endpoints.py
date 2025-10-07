"""
Monitoring endpoints for the Neural Showcase web API.

This module provides REST API endpoints for system monitoring,
health checks, and status dashboard.
"""

import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from monitoring.system_monitor import SystemMonitor
from monitoring.health_checks import get_health_checker, get_model_health_checker
from monitoring.logging_system import get_logging_system
from monitoring.dashboard import SystemDashboard, create_dashboard_api

# Initialize monitoring components
system_monitor = SystemMonitor(monitoring_interval=30)
health_checker = get_health_checker()
model_health_checker = get_model_health_checker()
logging_system = get_logging_system()
dashboard = SystemDashboard()

# Start monitoring
system_monitor.start_monitoring()
dashboard.start_monitoring()

# Create router
router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Dashboard API endpoints
dashboard_api = create_dashboard_api()


@router.get("/health")
async def health_check():
    """
    Get system health status.
    
    Returns comprehensive health check results for all system components.
    """
    try:
        overall_health = health_checker.get_overall_health()
        model_health = model_health_checker.check_all_models()
        
        # Combine results
        response = {
            "status": overall_health["overall_status"],
            "timestamp": datetime.now().isoformat(),
            "system_health": overall_health,
            "model_health": model_health,
            "summary": {
                "total_checks": overall_health["total_checks"] + len(model_health),
                "healthy": overall_health["status_counts"]["healthy"] + sum(1 for r in model_health.values() if r.status == "healthy"),
                "degraded": overall_health["status_counts"]["degraded"] + sum(1 for r in model_health.values() if r.status == "degraded"),
                "unhealthy": overall_health["status_counts"]["unhealthy"] + sum(1 for r in model_health.values() if r.status == "unhealthy")
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/health/simple")
async def simple_health_check():
    """
    Simple health check endpoint.
    
    Returns basic health status for load balancers and monitoring systems.
    """
    try:
        current_metrics = system_monitor.get_current_metrics()
        
        if not current_metrics:
            # Get metrics on-demand if monitoring is not active
            current_metrics = system_monitor._collect_system_metrics()
        
        # Simple health determination
        if (current_metrics.cpu_usage_percent > 95 or 
            current_metrics.memory_usage_percent > 95 or
            current_metrics.disk_usage_percent > 98):
            status = "unhealthy"
        elif (current_metrics.cpu_usage_percent > 85 or 
              current_metrics.memory_usage_percent > 85 or
              current_metrics.disk_usage_percent > 90):
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@router.get("/metrics")
async def get_system_metrics(hours: int = Query(default=1, ge=1, le=168)):
    """
    Get system performance metrics.
    
    Args:
        hours: Number of hours of historical data to include (1-168)
    
    Returns system metrics including CPU, memory, disk, and GPU usage.
    """
    try:
        current_metrics = system_monitor.get_current_metrics()
        metrics_summary = system_monitor.get_metrics_summary(hours=hours)
        
        return {
            "current": current_metrics,
            "summary": metrics_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/metrics/current")
async def get_current_metrics():
    """
    Get current system metrics.
    
    Returns the most recent system performance metrics.
    """
    try:
        current_metrics = system_monitor.get_current_metrics()
        
        if not current_metrics:
            # Get metrics on-demand if monitoring is not active
            current_metrics = system_monitor._collect_system_metrics()
        
        return {
            "metrics": current_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get current metrics: {str(e)}")


@router.get("/logs/summary")
async def get_log_summary(hours: int = Query(default=24, ge=1, le=168)):
    """
    Get logging summary.
    
    Args:
        hours: Number of hours of log data to summarize (1-168)
    
    Returns summary of log entries by level, module, and error patterns.
    """
    try:
        log_summary = logging_system.get_log_summary(hours=hours)
        
        return {
            "summary": log_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get log summary: {str(e)}")


@router.get("/dashboard")
async def get_dashboard():
    """
    Get complete system dashboard data.
    
    Returns comprehensive dashboard data including metrics, health, logs, and alerts.
    """
    try:
        dashboard_data = dashboard_api["dashboard"]()
        return dashboard_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


@router.get("/dashboard/status")
async def get_dashboard_status():
    """
    Get dashboard status summary.
    
    Returns a concise system status summary for quick overview.
    """
    try:
        status_summary = dashboard_api["status"]()
        return status_summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard status: {str(e)}")


@router.get("/alerts")
async def get_alerts():
    """
    Get active system alerts.
    
    Returns current system alerts and warnings.
    """
    try:
        alerts_data = dashboard_api["alerts"]()
        return alerts_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.post("/models/{model_name}/register")
async def register_model_for_monitoring(model_name: str):
    """
    Register a model for health monitoring.
    
    Args:
        model_name: Name of the model to register
    
    Note: This endpoint would typically receive model and sample input data,
    but for this demo it just acknowledges the registration request.
    """
    try:
        # In a real implementation, you would:
        # 1. Load the model
        # 2. Create appropriate sample input
        # 3. Register with model_health_checker.register_model()
        
        return {
            "message": f"Model '{model_name}' registration acknowledged",
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "note": "Full model registration requires model data and sample input"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register model: {str(e)}")


@router.get("/models/{model_name}/health")
async def get_model_health(model_name: str):
    """
    Get health status for a specific model.
    
    Args:
        model_name: Name of the model to check
    
    Returns health check results for the specified model.
    """
    try:
        model_result = model_health_checker.check_model_health(model_name)
        
        return {
            "model_name": model_name,
            "health": model_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check model health: {str(e)}")


@router.get("/models/health")
async def get_all_models_health():
    """
    Get health status for all registered models.
    
    Returns health check results for all models registered for monitoring.
    """
    try:
        all_models_health = model_health_checker.check_all_models()
        
        return {
            "models": all_models_health,
            "total_models": len(all_models_health),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check models health: {str(e)}")


@router.post("/inference/{model_name}")
async def record_inference_event(
    model_name: str,
    inference_time_ms: float,
    success: bool = True
):
    """
    Record a model inference event for monitoring.
    
    Args:
        model_name: Name of the model
        inference_time_ms: Inference time in milliseconds
        success: Whether the inference was successful
    
    This endpoint allows external systems to report inference events for monitoring.
    """
    try:
        # Record the inference event
        system_monitor.record_model_inference(model_name, inference_time_ms, success)
        
        # Also log the event
        logging_system.log_model_inference(model_name, inference_time_ms, success)
        
        return {
            "message": "Inference event recorded",
            "model_name": model_name,
            "inference_time_ms": inference_time_ms,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record inference event: {str(e)}")


@router.get("/system/info")
async def get_system_info():
    """
    Get basic system information.
    
    Returns system information including Python version, PyTorch version, and hardware info.
    """
    try:
        import platform
        import torch
        
        system_info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation()
            },
            "pytorch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Add GPU information if available
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "device_id": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / 1024**3,
                    "multi_processor_count": props.multi_processor_count
                })
            system_info["pytorch"]["gpu_devices"] = gpu_info
        
        return system_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")


# Cleanup function for graceful shutdown
def cleanup_monitoring():
    """Cleanup monitoring resources."""
    try:
        system_monitor.stop_monitoring()
        dashboard.stop_monitoring()
        logging_system.shutdown()
    except Exception as e:
        print(f"Error during monitoring cleanup: {e}")


# Export the router and cleanup function
__all__ = ["router", "cleanup_monitoring"]