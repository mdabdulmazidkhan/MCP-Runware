#!/usr/bin/env python3
"""
Runware API utilities and helper functions.
Provides functions for API communication, video model management, and video generation polling.
"""

import json
import os
import time
import uuid
from typing import Dict, Any, Optional, Tuple, List

import requests

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# API Configuration
DEFAULT_POLL_INTERVAL = 2
DEFAULT_TIMEOUT = 300
DEFAULT_API_BASE_URL = "https://api.runware.ai/v1"

# Video Models Information
VIDEO_MODELS: Dict[str, List[str]] = {
    "KlingAI": [
        "klingai:1@2 (KlingAI V1.0 Pro)",
        "klingai:1@1 (KlingAI V1 Standard)",
        "klingai:2@2 (KlingAI V1.5 Pro)",
        "klingai:2@1 (KlingAI V1.5 Standard)",
        "klingai:3@1 (KlingAI V1.6 Standard)",
        "klingai:3@2 (KlingAI V1.6 Pro)",
        "klingai:4@3 (KlingAI V2.1 Master)",
        "klingai:5@1 (KlingAI V2.1 Standard (I2V))",
        "klingai:5@2 (KlingAI V2.1 Pro (I2V))",
        "klingai:5@3 (KlingAI V2.0 Master)",
    ],
    "Veo": [
        "google:2@0 (Veo 2.0)",
        "google:3@0 (Veo 3.0)",
        "google:3@1 (Veo 3.0 Fast)",
    ],
    "Seedance": [
        "bytedance:2@1 (Seedance 1.0 Pro)",
        "bytedance:1@1 (Seedance 1.0 Lite)",
    ],
    "MiniMax": [
        "minimax:1@1 (MiniMax 01 Base)",
        "minimax:2@1 (MiniMax 01 Director)",
        "minimax:2@3 (MiniMax I2V 01 Live)",
        "minimax:3@1 (MiniMax 02 Hailuo)",
    ],
    "PixVerse": [
        "pixverse:1@1 (PixVerse v3.5)",
        "pixverse:1@2 (PixVerse v4)",
        "pixverse:1@3 (PixVerse v4.5)",
    ],
    "Vidu": [
        "vidu:1@0 (Vidu Q1 Classic)",
        "vidu:1@1 (Vidu Q1)",
        "vidu:1@5 (Vidu 1.5)",
        "vidu:2@0 (Vidu 2.0)",
    ],
    "Wan": [
        "runware:200@1 (Wan 2.1 1.3B)",
        "runware:200@2 (Wan 2.1 14B)",
    ],
}

# Model dimensions mapping
MODEL_DIMENSIONS: Dict[str, Dict[str, int]] = {
    # KlingAI Models
    "klingai:1@2": {"width": 1280, "height": 720},  # KlingAI V1.0 Pro
    "klingai:1@1": {"width": 1280, "height": 720},  # KlingAI V1 Standard
    "klingai:2@2": {"width": 1920, "height": 1080}, # KlingAI V1.5 Pro
    "klingai:2@1": {"width": 1280, "height": 720},  # KlingAI V1.5 Standard
    "klingai:3@1": {"width": 1280, "height": 720},  # KlingAI V1.6 Standard
    "klingai:3@2": {"width": 1920, "height": 1080}, # KlingAI V1.6 Pro
    "klingai:4@3": {"width": 1280, "height": 720},  # KlingAI V2.1 Master
    "klingai:5@1": {"width": 1280, "height": 720},  # KlingAI V2.1 Standard (I2V)
    "klingai:5@2": {"width": 1920, "height": 1080}, # KlingAI V2.1 Pro (I2V)
    "klingai:5@3": {"width": 1920, "height": 1080}, # KlingAI V2.0 Master
    
    # Veo Models
    "google:2@0": {"width": 1280, "height": 720},   # Veo 2.0
    "google:3@0": {"width": 1280, "height": 720},   # Veo 3.0
    "google:3@1": {"width": 1280, "height": 720},   # Veo 3.0 Fast
    
    # Seedance Models
    "bytedance:2@1": {"width": 864, "height": 480},  # Seedance 1.0 Pro
    "bytedance:1@1": {"width": 864, "height": 480},  # Seedance 1.0 Lite
    
    # MiniMax Models
    "minimax:1@1": {"width": 1366, "height": 768},  # MiniMax 01 Base
    "minimax:2@1": {"width": 1366, "height": 768},  # MiniMax 01 Director
    "minimax:2@3": {"width": 1366, "height": 768},  # MiniMax I2V 01 Live
    "minimax:3@1": {"width": 1366, "height": 768},  # MiniMax 02 Hailuo
    
    # PixVerse Models
    "pixverse:1@1": {"width": 640, "height": 360},  # PixVerse v3.5
    "pixverse:1@2": {"width": 640, "height": 360},  # PixVerse v4
    "pixverse:1@3": {"width": 640, "height": 360},  # PixVerse v4.5
    
    # Vidu Models
    "vidu:1@0": {"width": 1920, "height": 1080},    # Vidu Q1 Classic
    "vidu:1@1": {"width": 1920, "height": 1080},    # Vidu Q1
    "vidu:1@5": {"width": 1920, "height": 1080},    # Vidu 1.5
    "vidu:2@0": {"width": 1920, "height": 1080},    # Vidu 2.0
    
    # Wan Models
    "runware:200@1": {"width": 853, "height": 480}, # Wan 2.1 1.3B
    "runware:200@2": {"width": 853, "height": 480}, # Wan 2.1 14B
}


# ============================================================================
# CORE API FUNCTIONS
# ============================================================================

def genRandUUID() -> str:
    """
    Generate a random UUID string.
    
    Returns:
        A random UUID v4 string
        
    Example:
        >>> genRandUUID()
        '550e8400-e29b-41d4-a716-446655440000'
    """
    return str(uuid.uuid4())


def inferenceRequest(genConfig: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make an inference request to the Runware API.
    
    Args:
        genConfig: Configuration dictionary for the inference request
        
    Returns:
        Dict containing the API response
        
    Raises:
        ValueError: If required environment variables are missing
        Exception: If the request fails or returns an error
        
    Example:
        >>> config = {"taskType": "imageInference", "positivePrompt": "A cat"}
        >>> result = inferenceRequest(config)
    """
    # Get and validate environment variables
    api_key = os.getenv("RUNWARE_API_KEY")
    api_base_url = DEFAULT_API_BASE_URL
    session_timeout = DEFAULT_TIMEOUT
    
    if not api_key:
        raise ValueError("RUNWARE_API_KEY not found in environment variables")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate, br, zstd",
    }

    try:
        request_payload = [genConfig]
        response = requests.post(
            api_base_url,
            headers=headers,
            json=request_payload,
            timeout=session_timeout,
            allow_redirects=False,
            stream=True,
        )
        
        result = response.json()
        
        if "errors" in result:
            raise Exception(result["errors"][0]["message"])
            
        return result
        
    except Exception as e:
        raise Exception(f"Error: {e}")


# ============================================================================
# VIDEO MODEL MANAGEMENT FUNCTIONS
# ============================================================================

def getModelDimensions(model_id: str) -> Optional[Dict[str, int]]:
    """
    Get the supported dimensions for a specific video model.
    
    Args:
        model_id: The model identifier (e.g., "bytedance:2@1")
    
    Returns:
        Dictionary with "width" and "height" keys, or None if model not found
        
    Example:
        >>> getModelDimensions("klingai:1@2")
        {'width': 1280, 'height': 720}
        >>> getModelDimensions("nonexistent")
        None
    """
    return MODEL_DIMENSIONS.get(model_id)


def validateVideoDimensions(model_id: str, width: int, height: int) -> Tuple[bool, str]:
    """
    Validate if the provided dimensions are supported by the specified video model.
    
    Args:
        model_id: The model identifier
        width: Requested width
        height: Requested height
    
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> validateVideoDimensions("klingai:1@2", 1280, 720)
        (True, '')
        >>> validateVideoDimensions("klingai:1@2", 1920, 1080)
        (False, "Model 'klingai:1@2' only supports dimensions 1280x720, but you provided 1920x1080")
    """
    model_dims = getModelDimensions(model_id)
    if not model_dims:
        return False, f"Model '{model_id}' not found in supported video models"
    
    expected_width = model_dims["width"]
    expected_height = model_dims["height"]
    
    if width != expected_width or height != expected_height:
        return False, f"Model '{model_id}' only supports dimensions {expected_width}x{expected_height}, but you provided {width}x{height}"
    
    return True, ""


def getSupportedVideoModels() -> Dict[str, List[str]]:
    """
    Get all supported video models organized by provider.
    
    Returns:
        Dictionary of video models organized by provider
        
    Example:
        >>> models = getSupportedVideoModels()
        >>> models["KlingAI"]
        ['klingai:1@2 (KlingAI V1.0 Pro)', 'klingai:1@1 (KlingAI V1 Standard)', ...]
    """
    return VIDEO_MODELS


# ============================================================================
# VIDEO GENERATION POLLING FUNCTIONS
# ============================================================================

def pollVideoCompletion(taskUUID: str) -> Dict[str, Any]:
    """
    Poll for video generation completion using taskUUID.
    
    Args:
        taskUUID: The task UUID to poll for
        
    Returns:
        Dictionary containing the final result when video is ready or failed
        
    Raises:
        ValueError: If taskUUID is empty or API key is missing
        
    Example:
        >>> result = pollVideoCompletion("550e8400-e29b-41d4-a716-446655440000")
        >>> result["status"]
        'success'
    """
    # Input validation
    if not taskUUID:
        raise ValueError("taskUUID cannot be empty")
    
    if not os.getenv("RUNWARE_API_KEY"):
        raise ValueError("RUNWARE_API_KEY not found in environment variables")
    
    while True:
        time.sleep(DEFAULT_POLL_INTERVAL)
        
        poll_config = {
            "taskType": "getResponse",
            "taskUUID": taskUUID,
        }
        
        
        poll_result = inferenceRequest(poll_config)
        
        if "data" in poll_result and len(poll_result["data"]) > 0:
            video_data = poll_result["data"][0]
            status = video_data.get("status")
            
            if status != "processing":
                return poll_result



