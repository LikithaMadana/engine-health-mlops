"""
FastAPI inference service for engine health prediction models.
Supports both CNN and XGBoost models with batching for low-latency inference.
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
import torch
import numpy as np
import pickle
from typing import List, Optional
import os
import logging

# Import models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.cnn_model import EngineCNN
from src.models.xgboost_model import EngineXGBoost
from src.utils.exceptions import ModelLoadError, PredictionError, DataValidationError
from src.utils.validation import validate_feature_count, validate_probability
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__, log_file='logs/api.log')

app = FastAPI(
    title="Engine Health Prediction API",
    description="API for predicting engine health using CNN and XGBoost models",
    version="1.0.0"
)


# Request/Response models
class EngineData(BaseModel):
    """Single engine sensor reading."""
    features: List[float] = Field(..., description="Sensor readings (14 features)")
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 14:
            raise ValueError(f"Expected 14 features, got {len(v)}")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("All features must be numeric")
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("Features cannot contain NaN or Inf values")
        return v


class EngineDataBatch(BaseModel):
    """Batch of engine sensor readings."""
    samples: List[List[float]] = Field(..., description="Batch of sensor readings")
    
    @validator('samples')
    def validate_samples(cls, v):
        if len(v) == 0:
            raise ValueError("Batch cannot be empty")
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 samples")
        for i, sample in enumerate(v):
            if len(sample) != 14:
                raise ValueError(f"Sample {i}: expected 14 features, got {len(sample)}")
        return v


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: int = Field(..., description="Predicted class (0=healthy, 1=unhealthy)")
    probability: float = Field(..., description="Probability of predicted class")
    probabilities: List[float] = Field(..., description="Probabilities for all classes")
    
    @validator('prediction')
    def validate_prediction(cls, v):
        if v not in [0, 1]:
            raise ValueError(f"Prediction must be 0 or 1, got {v}")
        return v
    
    @validator('probability')
    def validate_probability_value(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {v}")
        return v


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: dict


# Global model instances
cnn_model = None
xgb_model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_models():
    """
    Load models at startup.
    
    Raises:
        ModelLoadError: If critical errors occur during model loading
    """
    global cnn_model, xgb_model
    
    logger.info("Starting model loading...")
    logger.info(f"Using device: {device}")
    
    # Load CNN model if available
    cnn_path = os.getenv('CNN_MODEL_PATH', 'models/cnn_model.pth')
    if os.path.exists(cnn_path):
        try:
            logger.info(f"Loading CNN model from {cnn_path}")
            cnn_model = EngineCNN(input_channels=14, window_size=50, hidden_channels=64)
            cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
            cnn_model.to(device)
            cnn_model.eval()
            logger.info(f"✓ CNN model loaded successfully from {cnn_path}")
        except FileNotFoundError:
            logger.warning(f"CNN model file not found: {cnn_path}")
        except Exception as e:
            logger.error(f"✗ Error loading CNN model: {e}")
            # Don't raise - allow API to start without CNN model
    else:
        logger.warning(f"CNN model path does not exist: {cnn_path}")
    
    # Load XGBoost model if available
    xgb_path = os.getenv('XGB_MODEL_PATH', 'models/xgb_model.pkl')
    if os.path.exists(xgb_path):
        try:
            logger.info(f"Loading XGBoost model from {xgb_path}")
            with open(xgb_path, 'rb') as f:
                xgb_model = pickle.load(f)
            logger.info(f"✓ XGBoost model loaded successfully from {xgb_path}")
        except FileNotFoundError:
            logger.warning(f"XGBoost model file not found: {xgb_path}")
        except Exception as e:
            logger.error(f"✗ Error loading XGBoost model: {e}")
            # Don't raise - allow API to start without XGBoost model
    else:
        logger.warning(f"XGBoost model path does not exist: {xgb_path}")
    
    # Log final status
    models_loaded = {
        "cnn": cnn_model is not None,
        "xgboost": xgb_model is not None
    }
    logger.info(f"Model loading complete. Status: {models_loaded}")


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("API startup initiated")
    load_models()
    logger.info("API ready to accept requests")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health status."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "cnn": cnn_model is not None,
            "xgboost": xgb_model is not None
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Returns health status and which models are loaded.
    """
    models_loaded = {
        "cnn": cnn_model is not None,
        "xgboost": xgb_model is not None
    }
    logger.debug(f"Health check: {models_loaded}")
    
    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded
    )


@app.post("/predict/cnn", response_model=PredictionResponse)
async def predict_cnn(data: EngineData):
    """
    Make prediction using CNN model.
    Expects windowed data (50 timesteps x 14 features).
    
    Args:
        data: EngineData with 14 sensor features
        
    Returns:
        PredictionResponse with prediction and probabilities
        
    Raises:
        HTTPException: 503 if model not loaded, 400 for invalid input, 500 for other errors
    """
    if cnn_model is None:
        logger.error("CNN prediction requested but model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CNN model not loaded. Please check server logs."
        )
    
    try:
        logger.debug(f"CNN prediction request received")
        
        # Reshape input for CNN: expecting (1, 50, 14)
        # For single prediction, we'll use the last 50 samples or pad if needed
        features = np.array(data.features, dtype=np.float32)
        
        # Validate input
        if len(features) != 14:
            raise DataValidationError(f"Expected 14 features, got {len(features)}")
        
        # Create a simple window (duplicate the single reading 50 times)
        # In production, you'd maintain a rolling window of historical data
        window = np.tile(features, (50, 1))
        
        # Convert to tensor
        x = torch.FloatTensor(window).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = cnn_model(x)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)
        
        pred_class = int(pred.item())
        probs_list = probs[0].cpu().numpy().tolist()
        
        logger.info(f"CNN prediction successful: class={pred_class}, "
                   f"confidence={probs_list[pred_class]:.4f}")
        
        return PredictionResponse(
            prediction=pred_class,
            probability=probs_list[pred_class],
            probabilities=probs_list
        )
    
    except DataValidationError as e:
        logger.warning(f"CNN prediction validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"CNN prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/xgboost", response_model=PredictionResponse)
async def predict_xgboost(data: EngineData):
    """
    Make prediction using XGBoost model.
    Expects flat feature vector (14 features).
    
    Args:
        data: EngineData with 14 sensor features
        
    Returns:
        PredictionResponse with prediction and probabilities
        
    Raises:
        HTTPException: 503 if model not loaded, 400 for invalid input, 500 for other errors
    """
    if xgb_model is None:
        logger.error("XGBoost prediction requested but model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="XGBoost model not loaded. Please check server logs."
        )
    
    try:
        logger.debug(f"XGBoost prediction request received")
        
        # Convert to numpy array
        features = np.array(data.features, dtype=np.float32).reshape(1, -1)
        
        # Validate input
        if features.shape[1] != 14:
            raise DataValidationError(f"Expected 14 features, got {features.shape[1]}")
        
        # Make prediction
        pred = xgb_model.predict(features)
        probs = xgb_model.predict_proba(features)
        
        pred_class = int(pred[0])
        probs_list = probs[0].tolist()
        
        logger.info(f"XGBoost prediction successful: class={pred_class}, "
                   f"confidence={probs_list[pred_class]:.4f}")
        
        return PredictionResponse(
            prediction=pred_class,
            probability=probs_list[pred_class],
            probabilities=probs_list
        )
    
    except DataValidationError as e:
        logger.warning(f"XGBoost prediction validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"XGBoost prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/cnn/batch", response_model=BatchPredictionResponse)
async def predict_cnn_batch(data: EngineDataBatch):
    """
    Make batch predictions using CNN model for low-latency inference.
    
    Args:
        data: EngineDataBatch with multiple samples
        
    Returns:
        BatchPredictionResponse with predictions for all samples
        
    Raises:
        HTTPException: 503 if model not loaded, 400 for invalid input, 500 for other errors
    """
    if cnn_model is None:
        logger.error("CNN batch prediction requested but model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CNN model not loaded"
        )
    
    try:
        batch_size = len(data.samples)
        logger.info(f"CNN batch prediction request: {batch_size} samples")
        
        # Process batch
        batch_windows = []
        for i, sample in enumerate(data.samples):
            features = np.array(sample, dtype=np.float32)
            if len(features) != 14:
                raise DataValidationError(f"Sample {i}: expected 14 features, got {len(features)}")
            window = np.tile(features, (50, 1))
            batch_windows.append(window)
        
        # Convert to tensor
        x = torch.FloatTensor(np.array(batch_windows)).to(device)
        
        # Make predictions
        with torch.no_grad():
            outputs = cnn_model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
        
        # Format results
        predictions = []
        for i in range(len(preds)):
            pred_class = int(preds[i].item())
            probs_list = probs[i].cpu().numpy().tolist()
            
            predictions.append(PredictionResponse(
                prediction=pred_class,
                probability=probs_list[pred_class],
                probabilities=probs_list
            ))
        
        logger.info(f"CNN batch prediction successful: {batch_size} samples processed")
        return BatchPredictionResponse(predictions=predictions)
    
    except DataValidationError as e:
        logger.warning(f"CNN batch validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"CNN batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.post("/predict/xgboost/batch", response_model=BatchPredictionResponse)
async def predict_xgboost_batch(data: EngineDataBatch):
    """
    Make batch predictions using XGBoost model for low-latency inference.
    
    Args:
        data: EngineDataBatch with multiple samples
        
    Returns:
        BatchPredictionResponse with predictions for all samples
        
    Raises:
        HTTPException: 503 if model not loaded, 400 for invalid input, 500 for other errors
    """
    if xgb_model is None:
        logger.error("XGBoost batch prediction requested but model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="XGBoost model not loaded"
        )
    
    try:
        batch_size = len(data.samples)
        logger.info(f"XGBoost batch prediction request: {batch_size} samples")
        
        # Convert to numpy array
        features = np.array(data.samples, dtype=np.float32)
        
        # Validate
        if features.shape[1] != 14:
            raise DataValidationError(f"Expected 14 features per sample, got {features.shape[1]}")
        
        # Make predictions
        preds = xgb_model.predict(features)
        probs = xgb_model.predict_proba(features)
        
        # Format results
        predictions = []
        for i in range(len(preds)):
            pred_class = int(preds[i])
            probs_list = probs[i].tolist()
            
            predictions.append(PredictionResponse(
                prediction=pred_class,
                probability=probs_list[pred_class],
                probabilities=probs_list
            ))
        
        logger.info(f"XGBoost batch prediction successful: {batch_size} samples processed")
        return BatchPredictionResponse(predictions=predictions)
    
    except DataValidationError as e:
        logger.warning(f"XGBoost batch validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"XGBoost batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
