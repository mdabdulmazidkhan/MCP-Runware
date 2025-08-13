"""
runware_mcp_server.py

This file implements a Runware MCP server using the SSE (Server-Sent Events) transport protocol.
It uses the FastMCP framework to expose tools that clients can call over an SSE connection.
SSE allows real-time, one-way communication from server to client over HTTP — ideal for pushing model updates.

The server uses:
- `Starlette` for the web server
- `uvicorn` as the ASGI server
- `FastMCP` from `mcp.server.fastmcp` to define the tools
- `SseServerTransport` to handle long-lived SSE connections
"""


#   [ MCP Client / Agent in Browser ]
#                  |
#      (connects via SSE over HTTP)
#                  |
#           [ Uvicorn Server ]
#                  |
#          (ASGI Protocol Bridge)
#                  |
#           [ Starlette App ]
#                  |
#           [ FastMCP Server ]
#                  |
#     @mcp.tool() like `imageInference`, `photoMaker`, `videoInference`, etc.
#                  |
#           [ Runware API ]

import os
import asyncio
import json
import base64
import time
import requests
from typing import TypedDict, Dict, Any, Optional, List, Union
from uuid import UUID
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server import Server  # Underlying server abstraction used by FastMCP
from mcp.server.sse import SseServerTransport  # The SSE transport layer

from starlette.applications import Starlette  # Web framework to define routes
from starlette.routing import Route, Mount  # Routing for HTTP and message endpoints
from starlette.requests import Request  # HTTP request objects

import uvicorn  # ASGI server to run the Starlette app

from utils.runwareUtils import inferenceRequest, genRandUUID, validateVideoDimensions, getModelDimensions, getSupportedVideoModels, pollVideoCompletion

load_dotenv()
mcp = FastMCP("Runware")

DEFAULT_IMAGE_MODEL = "civitai:943001@1055701"
DEFAULT_PHOTO_MAKER_MODEL = "civitai:139562@344487"
DEFAULT_BG_REMOVAL_MODEL = "runware:109@1"
DEFAULT_MASKING_MODEL = "runware:35@1"

def isClaudeUploadURL(url: str) -> bool:
    """Check if a URL is a Claude upload URL that should be rejected."""
    return isinstance(url, str) and url.startswith('https://files')

def validateRequiredParams(**kwargs) -> Optional[Dict[str, str]]:
    """Validate required parameters and return error dict if validation fails."""
    for param_name, param_value in kwargs.items():
        if not param_value:
            return {"status": "Tool error", "error": f"{param_name} is required"}
    return None

def validateImageInputs(**kwargs) -> Optional[Dict[str, str]]:
    """Validate image inputs and reject Claude upload URLs."""
    for param_name, param_value in kwargs.items():
        if param_value:
            if isinstance(param_value, str) and isClaudeUploadURL(param_value):
                return {"status": "Tool error", "error": "Pasting image will not work. Please provide the entire file path do not paste the image here."}
            elif isinstance(param_value, list):
                for item in param_value:
                    if isinstance(item, str) and isClaudeUploadURL(item):
                        return {"status": "Tool error", "error": "Pasting image will not work. Please provide the entire file path do not paste the image here."}
    return None


@mcp.tool()
async def imageInference(
    positivePrompt: str,
    model: str = DEFAULT_IMAGE_MODEL,
    height: Optional[int] = 1024,
    width: Optional[int] = 1024,
    numberResults: Optional[int] = 1,
    steps: Optional[int] = 20,
    CFGScale: Optional[float] = None,
    negativePrompt: Optional[str] = None,
    seed: Optional[int] = None,
    scheduler: Optional[str] = None,
    outputType: Optional[str] = None,
    outputFormat: Optional[str] = None,
    checkNSFW: Optional[bool] = None,
    strength: Optional[float] = None,
    clipSkip: Optional[int] = None,
    promptWeighting: Optional[str] = None,
    includeCost: Optional[bool] = None,
    vae: Optional[str] = None,
    maskMargin: Optional[int] = None,
    outputQuality: Optional[int] = None,
    taskUUID: Optional[UUID] = None,
    uploadEndpoint: Optional[str] = None,
    seedImage: Optional[str] = None,
    referenceImages: Optional[List[str]] = None,
    maskImage: Optional[str] = None,
    acceleratorOptions: Optional[Dict[str, Any]] = None,
    advancedFeatures: Optional[Dict[str, Any]] = None,
    controlNet: Optional[List[Dict[str, Any]]] = None,
    lora: Optional[List[Dict[str, Any]]] = None,
    lycoris: Optional[List[Dict[str, Any]]] = None,
    embeddings: Optional[List[Dict[str, Any]]] = None,
    ipAdapters: Optional[List[Dict[str, Any]]] = None,
    refiner: Optional[Dict[str, Any]] = None,
    outpaint: Optional[Dict[str, Any]] = None,
    instantID: Optional[Dict[str, Any]] = None,
    acePlusPlus: Optional[Dict[str, Any]] = None,
    extraArgs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate an image using Runware's image inference API with all available parameters.
    If user provides an image and asks to generate an image based on it, then use model "bytedance:4@1", and use seedImage parameter to pass the reference image.

    This function accepts all IImageInference parameters directly and generates images
    using the Runware API directly via HTTP requests. It supports the full range of parameters including basic
    settings, advanced features, and specialized configurations.

    Note: Display the url of the image inside the chat

    IMPORTANT: For image inputs (seedImage, referenceImages, maskImage), only accept:
    1. Publicly available URLs (e.g., "https://example.com/image.jpg")
    2. File paths that can be processed by imageUpload tool first
    3. Runware UUIDs from previously uploaded images
    
    Workflow: If user provides a local file path, first use imageUpload to get a Runware UUID, then use that UUID here.
    
    Args:
        positivePrompt (str): Text instruction to guide the model on generating the image, If you wish to generate an image without any prompt guidance, you can use the special token __BLANK__
        model (str): Model identifier (default: "civitai:943001@1055701")
        height (int): Image height (128-2048, divisible by 64, default: 1024)
        width (int): Image width (128-2048, divisible by 64, default: 1024)
        numberResults (int): Number of images to generate (1-20, default: 1). If user says "generate 4 images ..." then numberResults should be 4, says "create 2 images ... " then numberResults should be 2, etc.
        steps (int, optional): number of iterations the model will perform to generate the image (1-100, default: 20). The higher the number of steps, the more detailed the image will be
        CFGScale (float, optional): Represents how closely the images will resemble the prompt or how much freedom the AI model has (0-50, default: 7). Higher values are closer to the prompt. Low values may reduce the quality of the results.
        negativePrompt (str, optional): Negative guidance text. This parameter helps to avoid certain undesired results
        seed (int, optional): Random seed for reproducible results
        scheduler (str, optional): Inference scheduler. You can access list of available schedulers here https://runware.ai/docs/en/image-inference/schedulers
        outputType (str, optional): Specifies the output type in which the image is returned ('URL', 'dataURI', 'base64Data', default: 'URL')
        outputFormat (str, optional): Specifies the format of the output image ('JPG', 'PNG', 'WEBP', default: 'JPG')
        checkNSFW(bool, optional): Enable NSFW content check. When enabled, the API will check if the image contains NSFW (not safe for work) content. This check is done using a pre-trained model that detects adult content in images. (default: false)
        strength (float, optional): When doing image-to-image or inpainting, this parameter is used to determine the influence of the seedImage image in the generated output. A lower value results in more influence from the original image, while a higher value allows more creative deviation. (0-1, default: 0.8)
        clipSkip (int, optional): Defines additional layer skips during prompt processing in the CLIP model. Some models already skip layers by default, this parameter adds extra skips on top of those. (0-2)
        promptWeighting (str, optional): Prompt weighting method ('compel', 'sdEmbeds')
        includeCost (bool, optional): Include cost in response (default: false)
        vae (str, optional): VAE (Variational Autoencoder) model identifier
        maskMargin (int, optional): Adds extra context pixels around the masked region during inpainting (32-128)
        outputQuality (int, optional): Sets the compression quality of the output image. Higher values preserve more quality but increase file size, lower values reduce file size but decrease quality. (20-99, default: 95)
        taskUUID (UUID, optional): Unique task identifier
        uploadEndpoint (str, optional): Specifies a URL where the generated content will be automatically uploaded using the HTTP PUT method such as Cloud storage, Webhook services, CDN integration. The content data will be sent as the request body, allowing your endpoint to receive and process the generated image or video immediately upon completion.
        seedImage (str, optional): When doing image-to-image, inpainting or outpainting, this parameter is required. Specifies the seed image to be used for the diffusion process. ACCEPTS ONLY: Public URLs, Runware UUIDs, or file paths (use imageUpload first to get UUID). Supported formats are: PNG, JPG and WEBP
        referenceImages (List[str], optional): An array containing reference images used to condition the generation process. These images provide visual guidance to help the model generate content that aligns with the style, composition, or characteristics of the reference materials. ACCEPTS ONLY: Public URLs, Runware UUIDs, or file paths (use imageUpload first to get UUID).
        maskImage (str, optional): When doing inpainting, this parameter is required. Specifies the mask image to be used for the inpainting process. ACCEPTS ONLY: Public URLs, Runware UUIDs, or file paths (use imageUpload first to get UUID). Supported formats are: PNG, JPG and WEBP.
        acceleratorOptions (Dict[str, Any], optional): Advanced caching mechanisms to significantly speed up image generation by reducing redundant computation. teaCache  - {"teaCache": true  - Enables TeaCache for transformer-based models (e.g., Flux, SD 3) to accelerate iterative editing (default: false), "teaCacheDistance": 0.5  - Controls TeaCache reuse aggressiveness (0–1, default: 0.5); lower = better quality, higher = better speed} or deepCache- {"deepCache": true  - Enables DeepCache for UNet-based models (e.g., SDXL, SD 1.5) to cache internal feature maps for faster generation (default: false), "deepCacheInterval": 3  - Step interval between caching operations (min: 1, default: 3); higher = faster, lower = better quality, "deepCacheBranchId": 0  - Network branch index for caching depth (min: 0, default: 0); lower = faster, higher = more quality-preserving}
        advancedFeatures (Dict[str, Any], optional): Advanced generation features and is only available for the FLUX model architecture  "advancedFeatures": { "layerDiffuse": true}
        controlNet (List[Dict[str, Any]], optional): ControlNet provides a guide image to help the model generate images that align with the desired structure ControlNet configurations are "controlNet": [{"model": "string"  - ControlNet model ID (standard or AIR), "guideImage": "string"  - guide image (Public URLs, Runware UUIDs, or file paths - use imageUpload first to get UUID), "weight": 1.0  - strength of guidance (0–1, default 1), "startStep": 1  - step to start guidance, "endStep": 20  - step to end guidance, "startStepPercentage": 0  - alternative to startStep (0–99), "endStepPercentage": 100  - alternative to endStep (start+1–100), "controlMode": "balanced"  - guide vs. prompt priority ("prompt", "controlnet", "balanced")}]
        lora (List[Dict[str, Any]], optional): LoRA (Low-Rank Adaptation) to adapt a model to specific styles or features by emphasizing particular aspects of the data. model configurations "lora": [{"model": "string"  - AIR identifier of the LoRA model used to adapt style or features (e.g., "civitai:132942@146296"), "weight": 1.0  - Strength of the LoRA's influence (-4 to 4, default: 1); positive to apply style, negative to suppress it}]

        lycoris (List[Dict[str, Any]], optional): LyCORIS model configurations "lycoris {"model": model, "weight": weight}
        embeddings (List[Dict[str, Any]], optional): Textual inversion embeddings
        ipAdapters (List[Dict[str, Any]], optional):IP-Adapters enable image-prompted generation, allowing you to use reference images to guide the style and content of your generations. Multiple IP Adapters can be used simultaneously. IP-Adapter configurations "ipAdapters": [{"model": "string"  - AIR identifier of the IP-Adapter model used for image-based guidance (e.g., "runware:55@2"), "guideImage": "string"  - Reference image in Public URLs, Runware UUIDs, or file paths (use imageUpload first to get UUID) format (PNG/JPG/WEBP) to steer style/content, "weight": 1.0  - Influence strength (0–1, default: 1); 0 disables, 1 applies full guidance}]
        refiner (Dict[str, Any], optional): Refiner models help create higher quality image outputs by incorporating specialized models designed to enhance image details and overall coherence. Refiner model configuration "refiner": {"model": "string"  - AIR identifier of the SDXL-based refiner model (e.g., "civitai:101055@128080") used to enhance quality and detail, "startStep": 30  - Step at which the refiner begins processing (min: 2, max: total steps), or use "startStepPercentage" instead (1–99) for percentage-based control}

        outpaint (Dict[str, Any], optional): Outpainting configuration. Extends the image boundaries in specified directions. When using outpaint, you must provide the final dimensions using width and height parameters, which should account for the original image size plus the total extension (seedImage dimensions + top + bottom, left + right) "outpaint": {"top": 256  - Pixels to extend at the top (min: 0, multiple of 64), "right": 128  - Pixels to extend at the right (min: 0, multiple of 64), "bottom": 256  - Pixels to extend at the bottom (min: 0, multiple of 64), "left": 128  - Pixels to extend at the left (min: 0, multiple of 64), "blur": 16  - Blur radius (0–32, default: 0) to smooth transition between original and extended areas}

        instantID (Dict[str, Any], optional): InstantID configuration for identity-preserving image generation. "instantID": {"inputImage": "string" - Reference image for identity preservation (Public URLs, Runware UUIDs, or file paths - use imageUpload first to get UUID) in PNG/JPG/WEBP format, "poseImage": "string" - Pose reference image for pose guidance (Public URLs, Runware UUIDs, or file paths - use imageUpload first to get UUID) in PNG/JPG/WEBP format}
        acePlusPlus (Dict[str, Any], optional): acePlusPlus/ ACE++ for character-consistent generation. "acePlusPlus": {"type": "portrait"  - Task type ("portrait", "subject", "local_editing") for style or region-specific editing, "inputImages": ["string"]  - Reference image for identity/style preservation (Public URLs, Runware UUIDs, or file paths - use imageUpload first to get UUID), "inputMasks": ["string"]  - Mask image for targeted edits (white = edit, black = preserve), only used in local_editing, "repaintingScale": 0.5  - Controls balance between identity (0) and prompt adherence (1), default: 0}
        extraArgs (Dict[str, Any], optional): Extra arguments for the request
    
    Returns:
        dict: A dictionary containing the generation result with status, message, result data, parameters, and URL
        
    Example:
        >>> result = await imageInference(
        ...     positivePrompt="A beautiful sunset over mountains",
        ...     width=1024,
        ...     height=1024
        ... )
    """
    try:
        # Validate required parameters
        validation_error = validateRequiredParams(positivePrompt=positivePrompt)
        if validation_error:
            return validation_error
            
        # Validate image inputs
        validation_error = validateImageInputs(
            seedImage=seedImage,
            referenceImages=referenceImages,
            maskImage=maskImage
        )
        if validation_error:
            return validation_error
        
        params = {
            "taskType": "imageInference",
            "taskUUID" : taskUUID if taskUUID else genRandUUID(),
            "positivePrompt": positivePrompt,
            "model": model,
            "height": height,
            "width": width,
            "numberResults": numberResults
        }
        
        optional_params = {
            "steps": steps,
            "CFGScale": CFGScale,
            "negativePrompt": negativePrompt,
            "seed": seed,
            "scheduler": scheduler,
            "outputType": outputType,
            "outputFormat": outputFormat,
            "checkNSFW": checkNSFW,
            "strength": strength,
            "clipSkip": clipSkip,
            "promptWeighting": promptWeighting,
            "includeCost": includeCost,
            "vae": vae,
            "maskMargin": maskMargin,
            "outputQuality": outputQuality,
            "taskUUID": taskUUID,
            "uploadEndpoint": uploadEndpoint,
            "seedImage": seedImage,
            "referenceImages": referenceImages,
            "maskImage": maskImage,
            "acceleratorOptions": acceleratorOptions,
            "advancedFeatures": advancedFeatures,
            "controlNet": controlNet,
            "lora": lora,
            "lycoris": lycoris,
            "embeddings": embeddings,
            "ipAdapters": ipAdapters,
            "refiner": refiner,
            "outpaint": outpaint,
            "instantID": instantID,
            "acePlusPlus": acePlusPlus,
            "extraArgs": extraArgs
        }

        for key, value in optional_params.items():
            if value is not None:
                if key == "referenceImages":
                    if isinstance(value, str):
                        try:
                            if value.startswith('[') and value.endswith(']'):
                                params[key] = json.loads(value)
                            else:
                                params[key] = [value]
                        except json.JSONDecodeError:
                            params[key] = [value]
                    elif isinstance(value, list):
                        params[key] = value
                    else:
                        params[key] = [str(value)]
                
                else:
                    params[key] = value
        
        try:
            result = inferenceRequest(params)
            
            return {
                "status": "success",
                "message": "Image generation completed successfully",
                "result": result,
                }
        except Exception as e:
            
            return {"status": "API error", "message": str(e)}
            
    except Exception as e:
        return {"status": "Tool error", "error": str(e)}


@mcp.tool()
async def photoMaker(
    positivePrompt: str,
    inputImages: List[str],
    model: str = DEFAULT_PHOTO_MAKER_MODEL,  # RealVisXL V4.0
    height: Optional[int] = 1024,
    width: Optional[int] = 1024,
    style: Optional[str] = "No Style",
    strength: Optional[int] = 15,
    numberResults: Optional[int] = 1,
    steps: Optional[int] = 20,
    CFGScale: Optional[float] = 7.0,
    negativePrompt: Optional[str] = None,
    scheduler: Optional[str] = None,
    outputType: Optional[str] = None,
    outputFormat: Optional[str] = None,
    outputQuality: Optional[int] = 95,
    uploadEndpoint: Optional[str] = None,
    checkNSFW: Optional[bool] = None,
    includeCost: Optional[bool] = None,
    taskUUID: Optional[UUID] = None,
    clipSkip: Optional[int] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Transform and style images using PhotoMaker's advanced personalization technology.
    Create consistent, high-quality image variations with precise subject fidelity and style control.

    This function enables instant subject personalization without additional training. By providing up to
    four reference images, you can generate new images that maintain subject fidelity while applying
    various styles and compositions.

    IMPORTANT: For inputImages, only accept:
    1. Publicly available URLs (e.g., "https://example.com/image.jpg")
    2. File paths that can be processed by imageUpload tool first
    3. Runware UUIDs from previously uploaded images
    
    Workflow: If user provides a local file path, first use imageUpload to get a Runware UUID, then use that UUID here.

    Args:
        positivePrompt (str): Text instruction to guide the model (2-300 chars). The trigger word 'rwre' will be automatically prepended if not included in the prompt.
        inputImages (List[str]): 1-4 reference images of the subject. ACCEPTS ONLY: Public URLs, Runware UUIDs, or file paths (use imageUpload first to get UUID). Must contain clear faces for best results.
        model (str): SDXL-based model identifier (default: "civitai:139562@344487" - RealVisXL V4.0)
        height (int): Image height (128-2048, divisible by 64, default: 1024)
        width (int): Image width (128-2048, divisible by 64, default: 1024)
        style (str): Artistic style to apply ("No Style", "Cinematic", "Disney Character", "Digital Art", "Photographic", "Fantasy art", "Neonpunk", "Enhance", "Comic book", "Lowpoly", "Line art")
        strength (int): Balance between subject fidelity and transformation (15-50, default: 15). Lower values provide stronger subject fidelity.
        numberResults (int): Number of images to generate (1-20, default: 1)
        steps (int): Number of inference iterations (1-100, default: 20)
        CFGScale (float): How closely images match the prompt (0-50, default: 7)
        negativePrompt (str, optional): Text to guide what to avoid in generation
        scheduler (str, optional): Inference scheduler name
        outputType (str, optional): Output format ('URL', 'dataURI', 'base64Data', default: 'URL')
        outputFormat (str, optional): Image format ('JPG', 'PNG', 'WEBP', default: 'JPG')
        outputQuality (int, optional): Output image quality (20-99, default: 95)
        uploadEndpoint (str, optional): URL for automatic upload of generated content
        checkNSFW (bool, optional): Enable NSFW content check
        includeCost (bool, optional): Include generation cost in response
        taskUUID (UUID, optional): Unique task identifier
        clipSkip (int, optional): Additional CLIP model layer skips (0-2)
        seed (int, optional): Random seed for reproducible results

    Returns:
        dict: A dictionary containing the generation result with status, message, result data,
        parameters, and both image data for direct display and URLs.
        
    Example:
        >>> result = await photoMaker(
        ...     positivePrompt="A professional headshot",
        ...     inputImages=["path/to/reference.jpg"],
        ...     style="Photographic"
        ... )
    """
    try:
        # Validate required parameters
        validation_error = validateRequiredParams(
            positivePrompt=positivePrompt,
            inputImages=inputImages
        )
        if validation_error:
            return validation_error
            
        # Validate image inputs
        validation_error = validateImageInputs(inputImages=inputImages)
        if validation_error:
            return validation_error
        
        # Ensure rwre trigger word is in prompt
        if "rwre" not in positivePrompt:
            positivePrompt = f"rwre, {positivePrompt}"

        # Create params dict with required fields
        params = {
            "taskType": "photoMaker",
            "taskUUID": taskUUID if taskUUID else genRandUUID(),
            "positivePrompt": positivePrompt,
            "inputImages": inputImages,
            "model": model,
            "height": height,
            "width": width,
            "style": style,
            "strength": strength,
            "numberResults": numberResults,
            "steps": steps
        }

        # Add optional parameters if they are not None
        optional_params = {
            "CFGScale": CFGScale,
            "negativePrompt": negativePrompt,
            "scheduler": scheduler,
            "outputType": outputType,
            "outputFormat": outputFormat,
            "outputQuality": outputQuality,
            "uploadEndpoint": uploadEndpoint,
            "checkNSFW": checkNSFW,
            "includeCost": includeCost,
            "clipSkip": clipSkip,
            "seed": seed
        }

        for key, value in optional_params.items():
            if value is not None:
                params[key] = value

        # Call the inferenceRequest function with the parameters
        try:
            result = inferenceRequest(params)

            return {
                "status": "success",
                "message": "PhotoMaker image generation completed successfully",
                "result": result
            }

        except Exception as e:
            return {"status": "API error", "message": str(e)}

    except Exception as e:
        return {"status": "Tool error", "error": str(e)}


@mcp.tool()
async def imageUpscale(
    inputImage: str,
    upscaleFactor: int = 2,
    outputType: Optional[str] = None,
    outputFormat: Optional[str] = None,
    outputQuality: Optional[int] = 95,
    includeCost: Optional[bool] = None,
    taskUUID: Optional[UUID] = None
) -> dict:
    """
    Enhance the resolution and quality of images using Runware's advanced upscaling API.
    Transform low-resolution images into sharp, high-definition visuals.

    This function enables high-quality image upscaling with support for various input formats
    and flexible output options. The maximum output size is 4096x4096 pixels - larger inputs
    will be automatically resized to maintain this limit.

    IMPORTANT: For inputImage, only accept:
    1. Publicly available URLs (e.g., "https://example.com/image.jpg")
    2. File paths that can be processed by imageUpload tool first
    3. Runware UUIDs from previously uploaded images
    
    Workflow: If user provides a local file path, first use imageUpload to get a Runware UUID, then use that UUID here.

    Args:
        inputImage (str): Image to upscale. ACCEPTS ONLY: Public URLs, Runware UUIDs, or file paths (use imageUpload first to get UUID). Supported formats: PNG, JPG, WEBP
        upscaleFactor (int): Level of upscaling (2-4). Each level multiplies image size by that factor.For example, factor 2 doubles the image size. (default: 2)
        outputType (str, optional): Output format ('URL', 'dataURI', 'base64Data', default: 'URL')
        outputFormat (str, optional): Image format ('JPG', 'PNG', 'WEBP', default: 'JPG').  Note: PNG required for transparency.
        outputQuality (int, optional): Output image quality (20-99, default: 95)
        includeCost (bool, optional): Include generation cost in response
        taskUUID (UUID, optional): Unique task identifier

    Returns:
        dict: A dictionary containing the upscaling result with status, message, result data,
        parameters, and both image data for direct display and URLs.

    Note:
        Maximum output size is 4096x4096. If input size * upscaleFactor would exceed this,
        the input is automatically resized first. Example: 2048x2048 with factor 4 is reduced
        to 1024x1024 before upscaling.
    """
    try:
        if inputImage.startswith(('https://files')):
            return {"status": "Tool error", "error": "Pasting image will not work. Please provide the entire file path do not paste the image here."}
        
        # Create params dict with required fields
        params = {
            "taskType": "imageUpscale",
            "taskUUID": taskUUID if taskUUID else genRandUUID(),
            "inputImage": inputImage,
            "upscaleFactor": upscaleFactor
        }

        # Add optional parameters if they are not None
        optional_params = {
            "outputType": outputType,
            "outputFormat": outputFormat,
            "outputQuality": outputQuality,
            "includeCost": includeCost
        }

        for key, value in optional_params.items():
            if value is not None:
                params[key] = value

        # Call the inferenceRequest function with the parameters
        try:
            result = inferenceRequest(params)

            return {
                "status": "success",
                "message": "Image upscaling completed successfully",
                "result": result
            }

        except Exception as e:
            return {"status": "API error", "message": str(e)}

    except Exception as e:
        return {"status": "Tool error", "error": str(e)}


@mcp.tool()
async def imageBackgroundRemoval(
    inputImage: str,
    model: str = "runware:109@1",  # RemBG 1.4
    outputType: Optional[str] = None,
    outputFormat: Optional[str] = "PNG",  # Default PNG for transparency
    outputQuality: Optional[int] = 95,
    includeCost: Optional[bool] = None,
    taskUUID: Optional[UUID] = None,
    settings: Optional[Dict[str, Any]] = None
) -> dict:
    """
    Remove backgrounds from images effortlessly using Runware's low-cost image editing API.
    Isolate subjects from their backgrounds, creating images with transparent backgrounds.

    This function enables high-quality background removal with support for various input formats
    and advanced settings like alpha matting for enhanced edge quality.

    IMPORTANT: For inputImage, only accept:
    1. Publicly available URLs (e.g., "https://example.com/image.jpg")
    2. File paths that can be processed by imageUpload tool first
    3. Runware UUIDs from previously uploaded images
    
    Workflow: If user provides a local file path, first use imageUpload to get a Runware UUID, then use that UUID here.

    Args:
        inputImage (str): Image to process. ACCEPTS ONLY: Public URLs, Runware UUIDs, or file paths (use imageUpload first to get UUID). Supported formats: PNG, JPG, WEBP
        model (str): Background removal model to use (default: "runware:109@1" - RemBG 1.4)
            Available models:
            - runware:109@1: RemBG 1.4
            - runware:110@1: Bria RMBG 2.0
            - runware:112@1: BiRefNet v1 Base
            - runware:112@2: BiRefNet v1 Base - COD
            - runware:112@3: BiRefNet Dis
            - runware:112@5: BiRefNet General
            - runware:112@6: BiRefNet General Resolution 512x512 FP16
            - runware:112@7: BiRefNet HRSOD DHU
            - runware:112@8: BiRefNet Massive TR DIS5K TR TES
            - runware:112@9: BiRefNet Matting
            - runware:112@10: BiRefNet Portrait
        outputType (str, optional): Output format ('URL', 'dataURI', 'base64Data', default: 'URL')
        outputFormat (str, optional): Image format ('JPG', 'PNG', 'WEBP', default: 'PNG')
        outputQuality (int, optional): Output image quality (20-99, default: 95)
        includeCost (bool, optional): Include generation cost in response
        taskUUID (UUID, optional): Unique task identifier
        settings (Dict[str, Any], optional): Advanced settings (RemBG 1.4 model only):
            - rgba: [r, g, b, a] Background color and transparency (default: [255, 255, 255, 0])
            - postProcessMask (bool): Enable mask post-processing (default: False)
            - returnOnlyMask (bool): Return only the mask instead of processed image (default: False)
            - alphaMatting (bool): Enable alpha matting for better edges (default: False)
            - alphaMattingForegroundThreshold (int): Foreground threshold 1-255 (default: 240)
            - alphaMattingBackgroundThreshold (int): Background threshold 1-255 (default: 10)
            - alphaMattingErodeSize (int): Edge smoothing size 1-255 (default: 10)

    Returns:
        dict: A dictionary containing the background removal result with status, message,
        result data, parameters, and both image data for direct display and URLs.
    """
    try:
        if inputImage.startswith(('https://files')):
            return {"status": "Tool error", "error": "Pasting image will not work. Please provide the entire file path do not paste the image here."}
        
        # Create params dict with required fields
        params = {
            "taskType": "imageBackgroundRemoval",
            "taskUUID": taskUUID if taskUUID else genRandUUID(),
            "inputImage": inputImage,
            "model": model
        }

        # Add optional parameters if they are not None
        optional_params = {
            "outputType": outputType,
            "outputFormat": outputFormat,
            "outputQuality": outputQuality,
            "includeCost": includeCost
        }

        for key, value in optional_params.items():
            if value is not None:
                params[key] = value

        # Add settings if provided
        if settings is not None:
            params["settings"] = settings

        # Call the inferenceRequest function with the parameters
        try:
            

            
            result = inferenceRequest(params)

            return {
                "status": "success",
                "message": "Background removal completed successfully",
                "result": result
            }

        except Exception as e:
            return {"status": "API error", "message": str(e)}

    except Exception as e:
        return {"status": "Tool error", "error": str(e)}


@mcp.tool()
async def imageCaption(
    inputImage: str,
    includeCost: Optional[bool] = None,
    taskUUID: Optional[UUID] = None
) -> dict:
    """
    Generate image descriptions using Runware's API. Analyzes images to produce accurate
    and concise captions that can be used to create additional images or provide detailed
    insights into visual content.

    This function enables AI-powered image analysis to generate descriptive text prompts
    from images. It's useful for understanding image content or generating prompts for
    further image creation.

    IMPORTANT: For inputImage, only accept:
    1. Publicly available URLs (e.g., "https://example.com/image.jpg")
    2. File paths that can be processed by imageUpload tool first
    3. Runware UUIDs from previously uploaded images
    
    Workflow: If user provides a local file path, first use imageUpload to get a Runware UUID, then use that UUID here.

    Args:
        inputImage (str): Image to analyze. ACCEPTS ONLY: Public URLs, Runware UUIDs, or file paths (use imageUpload first to get UUID). Supported formats: PNG, JPG, WEBP
        includeCost (bool, optional): Include generation cost in response
        taskUUID (UUID, optional): Unique task identifier

    Returns:
        dict: A dictionary containing the caption generation result with status, message,
        result data (including the generated text), and cost if requested.
    """
    try:
        if inputImage.startswith(('https://files')):
            return {"status": "Tool error", "error": "Pasting image will not work. Please provide the entire file path do not paste the image here."}
        
        # Create params dict with required fields
        params = {
            "taskType": "imageCaption",
            "taskUUID": taskUUID if taskUUID else genRandUUID(),
            "inputImage": inputImage
        }

        # Add optional parameters if they are not None
        if includeCost is not None:
            params["includeCost"] = includeCost

        # Call the inferenceRequest function with the parameters
        try:
            result = inferenceRequest(params)

            # Prepare the response
            response_data = {
                "status": "success",
                "message": "Image caption generated successfully",
                "result": result
            }

            # Extract the generated text if available
            if isinstance(result, dict) and "data" in result:
                if isinstance(result["data"], list) and len(result["data"]) > 0:
                    if isinstance(result["data"][0], dict) and "text" in result["data"][0]:
                        response_data["caption"] = result["data"][0]["text"]

            return response_data

        except Exception as e:
            return {"status": "API error", "message": str(e)}

    except Exception as e:
        return {"status": "Tool error", "error": str(e)}


@mcp.tool()
async def imageMasking(
    inputImage: str,
    model: str = "runware:35@1",  # face_yolov8n - Lightweight face detection
    confidence: Optional[float] = 0.25,
    maxDetections: Optional[int] = 6,
    maskPadding: Optional[int] = 4,
    maskBlur: Optional[int] = 4,
    outputType: Optional[str] = None,
    outputFormat: Optional[str] = None,
    outputQuality: Optional[int] = 95,
    uploadEndpoint: Optional[str] = None,
    includeCost: Optional[bool] = None,
    taskUUID: Optional[UUID] = None
) -> dict:
    """
    Generate precise masks automatically for faces, hands, and people using AI detection.
    Enhance your inpainting workflow with smart, automated masking features.

    This function provides intelligent detection and mask generation for specific elements
    in images, particularly optimized for faces, hands, and people. Built on advanced
    detection models, it enhances the inpainting workflow by automatically creating
    precise masks around detected elements.

    IMPORTANT: For inputImage, only accept:
    1. Publicly available URLs (e.g., "https://example.com/image.jpg")
    2. File paths that can be processed by imageUpload tool first
    3. Runware UUIDs from previously uploaded images
    
    Workflow: If user provides a local file path, first use imageUpload to get a Runware UUID, then use that UUID here.

    Args:
        inputImage (str): Image to process. ACCEPTS ONLY: Public URLs, Runware UUIDs, or file paths (use imageUpload first to get UUID). Supported formats: PNG, JPG, WEBP
        model (str): Detection model to use:
            Face Detection Models:
            - "runware:35@1" - face_yolov8n: Lightweight model for 2D/realistic face detection
            - "runware:35@2" - face_yolov8s: Enhanced face detection with improved accuracy
            - "runware:35@6" - mediapipe_face_full: Specialized for realistic face detection
            - "runware:35@7" - mediapipe_face_short: Optimized face detection with reduced complexity
            - "runware:35@8" - mediapipe_face_mesh: Advanced face detection with mesh mapping
            
            Specialized Face Features:
            - "runware:35@9" - mediapipe_face_mesh_eyes_only: Focused detection of eye regions
            - "runware:35@15" - eyes_mesh_mediapipe: Specialized eyes detection
            - "runware:35@13" - nose_mesh_mediapipe: Specialized nose detection
            - "runware:35@14" - lips_mesh_mediapipe: Specialized lips detection
            - "runware:35@10" - eyes_lips_mesh: Detection of eyes and lips areas
            - "runware:35@11" - nose_eyes_mesh: Detection of nose and eyes areas
            - "runware:35@12" - nose_lips_mesh: Detection of nose and lips areas
            
            Hand & Person Detection:
            - "runware:35@3" - hand_yolov8n: Specialized for 2D/realistic hand detection
            - "runware:35@4" - person_yolov8n-seg: Person detection and segmentation
            - "runware:35@5" - person_yolov8s-seg: Advanced person detection with higher precision
        confidence (float, optional): Confidence threshold (0-1, default: 0.25).
            Lower values detect more objects but may introduce false positives.
        maxDetections (int, optional): Maximum elements to detect (1-20, default: 6).
            Only highest confidence detections are included if limit exceeded.
        maskPadding (int, optional): Extend/reduce mask area by pixels (default: 4).
            Positive values create larger masks, negative values shrink masks.
        maskBlur (int, optional): Blur mask edges by pixels (default: 4).
            Creates smooth transitions between masked and unmasked regions.
        outputType (str, optional): Output format ('URL', 'dataURI', 'base64Data', default: 'URL')
        outputFormat (str, optional): Image format ('JPG', 'PNG', 'WEBP', default: 'JPG')
        outputQuality (int, optional): Output quality (20-99, default: 95)
        uploadEndpoint (str, optional): URL for automatic upload using HTTP PUT
        includeCost (bool, optional): Include generation cost in response
        taskUUID (UUID, optional): Unique task identifier

    Returns:
        dict: A dictionary containing the masking result with status, message, result data,
        and parameters.

    Note:
        Generated masks can be used directly in inpainting workflows. When using
        maskMargin parameter in inpainting, the model will zoom into masked areas
        for enhanced detail generation.
    """
    try:
        if inputImage.startswith(('https://files')):
            return {"status": "Tool error", "error": "Pasting image will not work. Please provide the entire file path do not paste the image here."}
        
        # Create params dict with required fields
        params = {
            "taskType": "imageMasking",
            "taskUUID": taskUUID if taskUUID else genRandUUID(),
            "inputImage": inputImage,
            "model": model
        }

        # Add optional parameters if they are not None
        optional_params = {
            "confidence": confidence,
            "maxDetections": maxDetections,
            "maskPadding": maskPadding,
            "maskBlur": maskBlur,
            "outputType": outputType,
            "outputFormat": outputFormat,
            "outputQuality": outputQuality,
            "uploadEndpoint": uploadEndpoint,
            "includeCost": includeCost
        }

        for key, value in optional_params.items():
            if value is not None:
                params[key] = value

        # Call the inferenceRequest function with the parameters
        try:
            result = inferenceRequest(params)

            return {
                "status": "success",
                "message": "Image masking completed successfully",
                "result": result
            }

        except Exception as e:
            return {"status": "API error", "message": str(e)}

    except Exception as e:
        return {"status": "Tool error", "error": str(e)}


@mcp.tool()
async def modelSearch(
    search: Optional[str] = None,
    tags: Optional[List[str]] = None,
    category: Optional[str] = None,
    type: Optional[str] = None,
    architecture: Optional[str] = None,
    conditioning: Optional[str] = None,
    visibility: str = "all",
    limit: int = 20,
    offset: int = 0,
    taskUUID: Optional[UUID] = None
) -> dict:
    """
    Search and discover AI models available in the Runware platform.
    
    This tool enables discovery of available models on the Runware platform, providing powerful search 
    and filtering capabilities. Whether exploring public models from the community or managing private 
    models within your organization, this API helps find the perfect model for any image generation task.
    
    Models discovered through this tool can be immediately used in image generation tasks by referencing 
    their AIR identifiers. This enables dynamic model selection in applications and helps discover new 
    models for specific artistic styles.
    
    Args:
        search (str, optional): Search term to filter models. The search is performed across multiple fields:
            - Model name as exact phrase (boost: 10)
            - Model AIR identifier with wildcard matching (boost: 5)
            - Model name with wildcard matching (boost: 5)
            - Model version (exact word matching)
            - Model tags (exact word matching)
            The search is case-insensitive and results are ordered by relevance.
        
        tags (List[str], optional): Filter models by matching any of the provided tags. Models that contain 
            at least one of these tags will be included in the results.
        
        category (str, optional): Filter models by their category:
            - "checkpoint": Base models that serve as the foundation for image generation
            - "lora": LoRA (Low-Rank Adaptation) models for specific styles or concepts
            - "lycoris": Alternative to LoRA models with different adaptation techniques
            - "controlnet": Models for guided image generation with specific conditions
            - "vae": Variational Autoencoders for improving image quality and details
            - "embeddings": Textual embeddings for adding new concepts to the model's vocabulary
        
        type (str, optional): Filter checkpoint models by their type (only applicable when category is "checkpoint"):
            - "base": Standard models for general image generation
            - "inpainting": Models for filling in or modifying parts of existing images
            - "refiner": Models that improve the quality and details of generated images
        
        architecture (str, optional): Filter models by their architecture:
            FLUX Models: "flux1s", "flux1d", "fluxpro", "fluxultra", "fluxkontextdev", "fluxkontextpro", "fluxkontextmax"
            Imagen Models: "imagen3", "imagen3fast", "imagen4preview", "imagen4ultra", "imagen4fast"
            HiDream Models: "hidreamfast", "hidreamdev", "hidreamfull"
            SD Models: "sd1x", "sdhyper", "sd1xlcm", "sdxl", "sdxllcm", "sdxldistilled", "sdxlhyper", "sdxllightning", "sdxlturbo", "sd3"
            Other: "pony"
        
        conditioning (str, optional): Filter ControlNet models by their conditioning type (only applicable when category is "controlnet"):
            Edge Detection: "blur", "canny", "hed", "lineart", "softedge"
            Spatial: "depth", "normal", "seg"
            Creative: "inpaint", "inpaintdepth", "pix2pix", "scribble", "sketch"
            Specialized: "gray", "lowquality", "openmlsd", "openpose", "outfit", "qrcode", "shuffle", "tile"
        
        visibility (str): Filter models by visibility status and ownership:
            - "public": Show only your organization's public models
            - "private": Show only your organization's private models
            - "all": Show both community models and all your organization's models (default)
        
        limit (int): Maximum number of items to return (1-100, default: 20). Used for pagination.
        offset (int): Number of items to skip in the result set (min: 0, default: 0). Used for pagination.
        taskUUID (UUID, optional): Unique task identifier
    
    Returns:
        dict: A dictionary containing the model search results with status, message, result data,
        and comprehensive model information including AIR identifiers for immediate use.
    
    Note:
        For optimal search performance, consider using specific filters to narrow results and 
        combining multiple criteria to find the most relevant models. Results are returned in 
        paginated format for efficient processing of large result sets.
    """
    try:
        # Create params dict with required fields
        params = {
            "taskType": "modelSearch",
            "taskUUID": taskUUID if taskUUID else genRandUUID(),
            "visibility": visibility,
            "limit": limit,
            "offset": offset
        }
        
        optional_params = {
            "search": search,
            "tags": tags,
            "category": category,
            "type": type,
            "architecture": architecture,
            "conditioning": conditioning
        }
        
        for key, value in optional_params.items():
            if value is not None:
                params[key] = value
        
        try:
            result = inferenceRequest(params)
            
            return {
                "status": "success",
                "message": "Model search completed successfully",
                "result": result
            }
            
        except Exception as e:
            return {"status": "API error", "message": str(e)}
        
    except Exception as e:
        return {"status": "Tool error", "error": str(e)}


@mcp.tool()
async def videoInference(
    positivePrompt: str,
    model: Optional[str] = None,
    duration: Optional[float] = 5.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    outputType: Optional[str] = None,
    outputFormat: Optional[str] = None,
    uploadEndpoint: Optional[str] = None,
    includeCost: Optional[bool] = None,
    negativePrompt: Optional[str] = None,
    frameImages: Optional[Union[List[Dict[str, Any]], str]] = None,
    referenceImages: Optional[Union[List[str], str]] = None,
    fps: Optional[int] = None,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    CFGScale: Optional[float] = None,
    numberResults: Optional[int] = None,
    providerSettings: Optional[Dict[str, Any]] = None,
    deliveryMethod: str = "async",
    taskUUID: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate videos from text prompts and/or reference images using Runware's video generation API.
    
    Model Recommendations:
    - For Image-to-Video (I2V): Use 'klingai:5@2' if model is not provided, and use frameImages to guide the video generation, and do not use steps, CFGScale, or numberResults.
    - For Text-to-Video (T2V): Use 'google:3@1' if model is not provided, for pure text-based generation
    
    Args:
        positivePrompt: Text description of the video to generate
        model: Video generation model ID (recommended: klingai:5@2 ("width": 1920, "height": 1080) for I2V, google:3@1 ("width": 1280, "height": 720) for T2V)
        duration: The length of the generated video in seconds (min: 1, max: 10). This parameter directly affects the total number of frames produced based on the specified frame rate. Total frames are calculated as duration × fps. For example, a 5-second video at 24 fps will contain 120 frames. Longer durations require significantly more processing time and computational resources. Consider your specific use case when choosing duration length.
        width: Video width in pixels (optional, will be validated against model requirements)
        height: Video height in pixels (optional, will be validated against model requirements)
        outputType: Specifies the output type in which the video is returned. Currently, only URL delivery is supported for video outputs( default: "url")
        outputFormat: Specifies the format of the output video. Supported formats are: MP4 and WEBM (default: "mp4")
        outputQuality: Sets the compression quality of the output video. Higher values preserve more quality but increase file size, lower values reduce file size but decrease quality. (default: 95)
        uploadEndpoint: uploadEndpoint (str, optional): Specifies a URL where the generated content will be automatically uploaded using the HTTP PUT method such as Cloud storage, Webhook services, CDN integration. The content data will be sent as the request body, allowing your endpoint to receive and process the generated image or video immediately upon completion.
        includeCost: Whether to include cost information in response (default: False)
        negativePrompt: Text describing what NOT to include in the video. Common negative prompts for video include terms like "blurry", "low quality", "distorted", "static", "flickering", or specific content you want to exclude.
        frameImages: Array of frame objects that define key frames to guide video generation. Each object specifies an input image and optionally its position within the video timeline. This allows constraining specific frames within the video sequence, ensuring particular visual content appears at designated points (different from referenceImages which provide overall visual guidance without timeline constraints).
        
            Frame positioning options:
            - Omit frame parameter: Automatic distribution applies
            * 1 image: Used as first frame
            * 2 images: First and last frames  
            * 3+ images: First, last, and evenly spaced intermediate frames
            - Named positions: "first" or "last"
            - Numeric positions: 0 (first frame) or any positive integer within frame count
            
            Example structures:
            - Single frame: [{"inputImage": "uuid_or_url"}]
            - First/last: [{"inputImage": "uuid1", "frame": "first"}, {"inputImage": "uuid2", "frame": "last"}]
            - Mixed: [{"inputImage": "uuid1", "frame": 0}, {"inputImage": "uuid2", "frame": 48}, {"inputImage": "uuid3", "frame": "last"}]
            
            inputImage accepts: UUID strings, data URIs, base64 data, or public URLs (PNG/JPG/WEBP)
            
        referenceImages: Array containing reference images used to condition the generation process. These images provide visual guidance to help the model generate content that aligns with the style, composition, or characteristics of the reference materials. Unlike frameImages which constrain specific timeline positions, reference images guide the general appearance that should appear consistently across the video. Reference images work in combination with your text prompt to provide both textual and visual guidance for the generation process.
        
            Each image can be specified in one of the following formats:
            - UUID v4 string of a previously uploaded image or generated image
            - Data URI string in format: data:<mediaType>;base64,<base64_data>
            - Base64 encoded image without data URI prefix
            - Public URL pointing to the image (PNG/JPG/WEBP supported)
            
            Example: ["aac49721-1964-481a-ae78-8a4e29b91402"] or ["https://example.com/image.jpg"]
        
        fps: The frame rate (frames per second) of the generated video. Higher frame rates create smoother motion but require more processing time and result in larger file sizes. example: 24 fps, 30 fps, 60 fps (default: 24 fps). Note: Using the same duration with higher frame rates creates smoother motion by generating more intermediate frames. The frame rate combines with duration to determine total frame count: duration × fps = total frames.
        
        steps: The number of denoising steps the model performs during video generation. More steps typically result in higher quality output but require longer processing time.
    
        seed: Random seed for reproducible results
        
        CFGScale: Controls how closely the video generation follows your prompt. Higher values make the model adhere more strictly to your text description, while lower values allow more creative freedom. range: 0 - 50.0
        
        numberResults: Specifies how many videos to generate for the given parameters (default: 1)
        
        providerSettings: Additional provider-specific settings
        
        deliveryMethod: Determines how the video generation results are delivered. Currently, video inference only supports asynchronous processing due to the computational intensity of video generation. (default: "async")
        
        taskUUID: Custom task UUID (auto-generated if not provided)
    
    Returns:
        Dictionary containing video generation status and results
        
    Note:
        - Width and height will be automatically validated against the selected model's supported dimensions
        - For image-to-video generation, provide referenceImages or frameImages
        - For text-to-video generation, only positivePrompt is required
        - The tool automatically polls for completion when deliveryMethod is "async"
    """
    try:
        if referenceImages:
            for img in referenceImages:
                if isinstance(img, str) and img.startswith('https://files'):
                    return {"status": "Tool error", "error": "Pasting image will not work. Please provide the entire file path do not paste the image here."}
        
        if frameImages:
            for frame in frameImages:
                if isinstance(frame, dict) and 'inputImages' in frame:
                    input_imgs = frame['inputImages']
                    if isinstance(input_imgs, list):
                        for img in input_imgs:
                            if isinstance(img, str) and img.startswith('https://files'):
                                return {"status": "Tool error", "error": "Pasting image will not work. Please provide the entire file path do not paste the image here."}
                    elif isinstance(input_imgs, str) and input_imgs.startswith('https://files'):
                        return {"status": "Tool error", "error": "Pasting image will not work. Please provide the entire file path do not paste the image here."}
        
        if width is not None and height is not None:
            is_valid, error_msg = validateVideoDimensions(model, width, height)
            if not is_valid:
                model_dims = getModelDimensions(model)
                if model_dims:
                    supported_dims = f"{model_dims['width']}x{model_dims['height']}"
                    return {
                        "status": "Tool error", 
                        "error": f"{error_msg}\n\nSupported dimensions for {model}: {supported_dims}\n\nPlease adjust your width and height parameters accordingly."
                    }
                else:
                    return {
                        "status": "Tool error", 
                        "error": f"{error_msg}\n\nThis model may not be supported or may have different dimension requirements."
                    }
        
        params = {
            "taskType": "videoInference",
            "taskUUID": taskUUID if taskUUID else genRandUUID(),
            "positivePrompt": positivePrompt,
            "model": model,
            "duration": duration,
            "deliveryMethod": deliveryMethod
        }
        
        optional_params = {
            "width": width,
            "height": height,
            "outputType": outputType,
            "outputFormat": outputFormat,
            "uploadEndpoint": uploadEndpoint,
            "includeCost": includeCost,
            "negativePrompt": negativePrompt,
            "frameImages": frameImages,
            "referenceImages": referenceImages,
            "fps": fps,
            "steps": steps,
            "seed": seed,
            "CFGScale": CFGScale,
            "numberResults": numberResults,
            "providerSettings": providerSettings
        }
        
        for key, value in optional_params.items():
            if value is not None:
                if key == "referenceImages":
                    if isinstance(value, str):
                        try:
                            if value.startswith('[') and value.endswith(']'):
                                params[key] = json.loads(value)
                            else:
                                params[key] = [value]
                        except json.JSONDecodeError:
                            params[key] = [value]
                    elif isinstance(value, list):
                        params[key] = value
                    else:
                        params[key] = [str(value)]
                if key == "frameImages":
                    if isinstance(value, str):
                        try:
                            if value.startswith('[') and value.endswith(']'):
                                params[key] = json.loads(value)
                            else:
                                params[key] = [value]
                        except json.JSONDecodeError:
                            params[key] = [value]
                    elif isinstance(value, list):
                        params[key] = value
                    else:
                        params[key] = [str(value)]
                else:
                    params[key] = value
        
        try:
            result = inferenceRequest(params)
            
            if not result or not isinstance(result, dict):
                return {
                    "status": "error",
                    "message": "Invalid response from API",
                    "result": result
                }
            
            taskUUID = None
            if "data" in result and isinstance(result["data"], list) and len(result["data"]) > 0:
                first_item = result["data"][0]
                if isinstance(first_item, dict) and "taskUUID" in first_item:
                    taskUUID = first_item["taskUUID"]
            
            if not taskUUID:
                return {
                    "status": "error",
                    "message": "No taskUUID received from API",
                    "result": result
                }
            
            return pollVideoCompletion(taskUUID)
            
        except Exception as e:
            return {"status": "API error", "error": str(e)}
        
    except Exception as e:
        return {"status": "Tool error", "error": str(e)}


@mcp.tool()
async def listVideoModels() -> dict:
    """
    List all available video models and their supported dimensions.
    
    This tool provides a comprehensive overview of all supported video generation models
    on the Runware platform, organized by provider with their specific dimension requirements.
    
    Returns:
        dict: A dictionary containing all supported video models organized by provider,
        with their identifiers and supported dimensions.
    
    Note:
        Use the returned model identifiers directly in the videoInference tool.
        Each model has specific dimension requirements that must be followed.
    """
    try:
        video_models = getSupportedVideoModels()
        
        response = {
            "status": "success",
            "message": "Video models retrieved successfully",
            "models": video_models,
            "total_providers": len(video_models),
            "usage_note": "Use these model identifiers directly in videoInference tool"
        }
        
        model_details = {}
        for provider, models in video_models.items():
            model_details[provider] = []
            for model_info in models:
                model_id = model_info.split(" (")[0]
                dimensions = getModelDimensions(model_id)
                if dimensions:
                    model_details[provider].append({
                        "model_id": model_id,
                        "display_name": model_info,
                        "supported_dimensions": f"{dimensions['width']}x{dimensions['height']}",
                        "width": dimensions["width"],
                        "height": dimensions["height"]
                    })
        
        response["model_details"] = model_details
        
        return response
        
    except Exception as e:
        return {"status": "Tool error", "error": str(e)}


@mcp.tool()
async def getVideoModelInfo(model_id: str) -> dict:
    """
    Get detailed information about a specific video model including supported dimensions.
    
    This tool provides comprehensive information about a specific video generation model,
    including its supported dimensions, provider, and usage recommendations.
    
    Args:
        model_id (str): The model identifier (e.g., "bytedance:2@1", "klingai:1@2")
    
    Returns:
        dict: A dictionary containing detailed model information including supported dimensions,
        provider details, and usage recommendations.
    
    Note:
        Use the returned dimensions in your videoInference requests to avoid errors.
    """
    try:
        dimensions = getModelDimensions(model_id)
        
        if not dimensions:
            return {
                "status": "error",
                "message": f"Model '{model_id}' not found in supported video models",
                "suggestion": "Use listVideoModels() to see all available models"
            }
        
        video_models = getSupportedVideoModels()
        provider_info = None
        display_name = None
        
        for provider, models in video_models.items():
            for model_info in models:
                if model_info.startswith(model_id):
                    provider_info = provider
                    display_name = model_info
                    break
            if provider_info:
                break
        
        response = {
            "status": "success",
            "message": f"Model information retrieved for {model_id}",
            "model_id": model_id,
            "display_name": display_name or model_id,
            "provider": provider_info or "Unknown",
            "supported_dimensions": {
                "width": dimensions["width"],
                "height": dimensions["height"],
                "format": f"{dimensions['width']}x{dimensions['height']}"
            }
        }
        
        return response
        
    except Exception as e:
        return {"status": "Tool error", "error": str(e)}


@mcp.tool()
async def imageUpload(file_path: str) -> dict:
    """
    Upload an image to Runware by providing a file path on the local file system or publicurl.
    
    This function reads the image file directly from the file path or public url and uploads it to Runware.
    It returns a Runware UUID that can be used in other tools that require image inputs.
    
    IMPORTANT: This tool should be used FIRST when you need to process local images with other Runware tools.
    The returned UUID can then be used as input for tools like imageInference, photoMaker, imageUpscale, etc.
    
    Workflow:
    1. Use this tool to upload a local image and get a Runware UUID
    2. Use the returned UUID in other Runware tools that require image inputs
    3. This prevents context pollution and ensures proper image handling

    Args:
        file_path (str): Path to the image file on the local file system
            Examples: "/path/to/image.jpg", "images/photo.png", "./uploads/file.webp"
            Supported formats: PNG, JPG, WEBP, JPEG, BMP, GIF
    
    Returns:
        dict: A dictionary containing the upload result with status, message, result data,
        and the uploaded image UUID for future reference.
    
    Note:
        - Images are deleted 30 days after last use, but remain available indefinitely while in use
        - File path must be accessible
        - Use the returned UUID in other Runware tools instead of file paths or base64 data
    """
    try:
        if not file_path or not isinstance(file_path, str):
            return {"status": "error", "message": "Invalid file path provided"}
        if file_path.startswith(('https://files')):
            return {
                "status": "error", 
                "message": f"Please provide file path, do not paste the image here."
            }

        if not file_path.endswith(('.jpg', '.jpeg', '.png', '.webp')):
            return {
                "status": "error", 
                "message": f"You provided {file_path}\n\nThis tool only accepts .jpg, .jpeg, .png, .webp files. Do not paste the image here."
            }
        
        if not os.path.exists(file_path):
            return {
                "status": "error", 
                "message": f"File not found: {file_path}\n\nPlease check:\n1. The file path is correct and complete\n2. The file exists in the specified location\n3. You have permission to access the file\n\nExamples of correct file paths:\n- '/Users/username/Pictures/image.jpg'\n- './images/photo.png'\n- 'C:\\Users\\username\\Desktop\\image.jpg'"
            }
        
        if not os.path.isfile(file_path):
            return {"status": "error", "message": f"Path is not a file: {file_path}\n\nPlease provide a path to an image file, not a directory. Do not paste the image here."}
        
        try:
            with open(file_path, 'rb') as file:
                file_data = file.read()
                base64_data = base64.b64encode(file_data).decode('utf-8')
        except Exception as e:
            return {"status": "error", "message": f"Failed to read file: {str(e)}\n\nPlease check if the file is accessible and not corrupted. Do not paste the image here."}
        
        params = {
            "taskType": "imageUpload",
            "taskUUID": genRandUUID(),
            "image": base64_data
        }
        
        try:
            result = inferenceRequest(params)
            
            return {
                "status": "success",
                "message": f"Image uploaded successfully from file: {file_path}",
                "result": result
            }
            
        except Exception as e:
            return {"status": "API error", "message": str(e)}
        
    except Exception as e:
        return {"status": "Tool error", "error": str(e)}



def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """
    Constructs a Starlette app with SSE and message endpoints.

    Args:
        mcp_server (Server): The core MCP server instance.
        debug (bool): Enable debug mode for verbose logs.

    Returns:
        Starlette: The full Starlette app with routes.
    """
    # Create SSE transport handler to manage long-lived SSE connections
    sse = SseServerTransport("/messages/")

    # This function is triggered when a client connects to `/sse`
    async def handle_sse(request: Request) -> None:
        """
        Handles a new SSE client connection and links it to the MCP server.
        """
        # Open an SSE connection, then hand off read/write streams to MCP
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,  # Low-level send function provided by Starlette
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    # Return the Starlette app with configured endpoints
    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),          # For initiating SSE connection
            Mount("/messages/", app=sse.handle_post_message),  # For POST-based communication
        ],
    )

if __name__ == "__main__":
    # Get the underlying MCP server instance from FastMCP
    mcp_server = mcp._mcp_server  # Accessing private member (acceptable here)

    # Command-line arguments for host/port control
    import argparse

    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8081, help='Port to listen on')
    args = parser.parse_args()

    # Build the Starlette app with debug mode enabled
    starlette_app = create_starlette_app(mcp_server, debug=True)

    # Launch the server using Uvicorn
    uvicorn.run(starlette_app, host=args.host, port=args.port)
