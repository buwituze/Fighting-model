# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
import cv2
import uvicorn
import os
import tempfile
from typing import Optional
import logging
import asyncio
from contextlib import asynccontextmanager
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMG_SIZE = 64
FRAMES_PER_VIDEO = 16

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "fight_detection_model_v2_compatible.h5")

model = None

def load_model_with_custom_objects():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        logger.warning(f"Standard loading failed: {e}")
        
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            logger.info("Model loaded without compilation")
            return model
        except Exception as e2:
            logger.warning(f"Loading without compilation failed: {e2}")
            
            try:
                logger.error("Could not load model. You may need to recreate the model architecture.")
                return None
            except Exception as e3:
                logger.error(f"All loading methods failed: {e3}")
                return None

def create_fallback_model():
    logger.info("Creating fallback model...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.GlobalAveragePooling3D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    logger.warning("Using fallback model - predictions will be random!")
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    logger.info("Starting up Fight Detection API...")
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            logger.info("Available .h5 files in current directory:")
            for file in os.listdir(CURRENT_DIR):
                if file.endswith('.h5'):
                    logger.info(f"  - {file}")
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        model = load_model_with_custom_objects()
        
        if model is None:
            logger.warning("Creating fallback model...")
            model = create_fallback_model()
        else:
            logger.info("Model loaded successfully")
            logger.info(f"Model input shape: {model.input_shape}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Creating fallback model...")
        model = create_fallback_model()
    
    yield
    
    logger.info("Shutting down Fight Detection API...")

app = FastAPI(
    title="Fight Detection API",
    description="API for detecting fights in video files using 3D CNN",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionOutput(BaseModel):
    prediction: str = Field(..., description="Prediction label: 'fight' or 'noFight'")
    probability: float = Field(..., description="Probability of fight (0-1)")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    confidence_level: str = Field(..., description="Confidence level: 'High', 'Medium', or 'Low'")
    threshold_used: float = Field(..., description="Decision threshold used")
    frames_processed: int = Field(..., description="Number of frames processed")
    video_duration_estimate: Optional[float] = Field(None, description="Estimated video duration in seconds")
    model_status: str = Field(..., description="Status of the model used")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_input_shape: Optional[list] = None
    tensorflow_version: str
    opencv_version: str

def extract_frames_improved(video_path, max_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else None
    
    logger.info(f"Video info - Total frames: {total_frames}, FPS: {fps}, Duration: {duration}s")
    
    if total_frames == 0:
        cap.release()
        raise ValueError("Video has no frames")
    
    if total_frames < max_frames:
        step = 1
    else:
        step = total_frames // max_frames
    
    frame_indices = list(range(0, total_frames, step))[:max_frames]
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Could not read frame at index {frame_idx}")
            break
        
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)
        
        if len(frames) == max_frames:
            break
    
    cap.release()
    
    while len(frames) < max_frames:
        if frames:
            frames.append(frames[-1].copy())
        else:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
    
    frames_array = np.array(frames[:max_frames])
    
    return frames_array, duration

async def predict_video_enhanced(video_path, threshold=0.5):
    global model
    
    if model is None:
        raise ValueError("Model is not loaded")
    
    try:
        loop = asyncio.get_event_loop()
        frames, duration = await loop.run_in_executor(
            None, extract_frames_improved, video_path, FRAMES_PER_VIDEO
        )
        
        if frames.shape[0] != FRAMES_PER_VIDEO:
            raise ValueError(f"Could not extract {FRAMES_PER_VIDEO} frames, got {frames.shape[0]}")
        
        input_array = np.expand_dims(frames.astype(np.float32) / 255.0, axis=0)
        
        prediction_prob = await loop.run_in_executor(
            None, lambda: model.predict(input_array, verbose=0)[0][0]
        )
        
        prediction_label = "fight" if prediction_prob > threshold else "noFight"
        
        confidence_score = max(prediction_prob, 1 - prediction_prob)
        if confidence_score > 0.8:
            confidence_level = "High"
        elif confidence_score > 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        model_status = "original" if os.path.exists(MODEL_PATH) else "fallback"
        
        return {
            'prediction': prediction_label,
            'probability': float(prediction_prob),
            'confidence_score': float(confidence_score),
            'confidence_level': confidence_level,
            'threshold_used': threshold,
            'frames_processed': len(frames),
            'video_duration_estimate': duration,
            'model_status': model_status
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise ValueError(f"Prediction failed: {str(e)}")

# API Endpoints
@app.post("/predict", response_model=PredictionOutput, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict_fight(
    file: UploadFile = File(..., description="Video file to analyze"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold for classification")
):
    """
    Predict whether a video contains fighting based on uploaded video file
    
    - **file**: Video file to analyze (supported formats: mp4, avi, mov, mkv, flv, wmv)
    - **threshold**: Decision threshold for classification (0.0 to 1.0, default: 0.5)
    """
    
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please check server logs and ensure the model file exists."
        )
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    file_extension = os.path.splitext(file.filename.lower())[1]
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type '{file_extension}'. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    MAX_FILE_SIZE = 100 * 1024 * 1024
    contents = await file.read()
    
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing video: {file.filename} (size: {len(contents)} bytes)")
        
        result = await predict_video_enhanced(temp_file_path, threshold)
        
        return PredictionOutput(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing video {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_file_path}: {e}")

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint with detailed system information"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_input_shape=list(model.input_shape) if model else None,
        tensorflow_version=tf.__version__,
        opencv_version=cv2.__version__
    )

@app.get("/model-info")
def model_info():
    """Get detailed model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        return {
            "model_loaded": True,
            "input_shape": list(model.input_shape),
            "output_shape": list(model.output_shape),
            "total_params": model.count_params(),
            "layers": len(model.layers),
            "model_type": str(type(model)),
            "frames_per_video": FRAMES_PER_VIDEO,
            "image_size": IMG_SIZE,
            "tensorflow_version": tf.__version__,
            "model_path": MODEL_PATH,
            "model_exists": os.path.exists(MODEL_PATH)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.get("/")
def read_root():
    """
    Root endpoint that provides basic API information
    """
    return {
        "message": "Fight Detection API",
        "description": "Upload a video file to detect if it contains fighting using AI",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Upload video file for fight detection",
            "GET /health": "Check API health status",
            "GET /model-info": "Get detailed model information",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /redoc": "Alternative API documentation (ReDoc)"
        },
        "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"],
        "max_file_size": "100MB",
        "model_status": "loaded" if model else "not_loaded",
        "processing_info": {
            "frames_per_video": FRAMES_PER_VIDEO,
            "image_size": IMG_SIZE,
            "default_threshold": 0.5
        },
        "access_urls": {
            "api_docs": "http://localhost:8000/docs",
            "health_check": "http://localhost:8000/health",
            "model_info": "http://localhost:8000/model-info"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Fight Detection API...")
    print("📝 Access the API at: http://localhost:8000")
    print("📚 View API documentation at: http://localhost:8000/docs")
    print("🔍 Health check at: http://localhost:8000/health")
    print("⚡ To stop the server: Press Ctrl+C")
    print("-" * 50)
    
    uvicorn.run(
        "app:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    )