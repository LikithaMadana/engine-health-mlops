"""
Example client for testing the inference API.

This script demonstrates how to use the FastAPI endpoints for predictions.
"""

import requests
import json
import numpy as np


def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()


def test_cnn_prediction():
    """Test CNN single prediction."""
    print("\nTesting CNN single prediction...")
    
    # Create sample data (14 features)
    sample_data = {
        "features": [0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.6, 0.4, 0.7, 0.3, 0.5, 0.2, 0.8, 0.4]
    }
    
    response = requests.post(
        "http://localhost:8000/predict/cnn",
        json=sample_data
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']} (0=Healthy, 1=Unhealthy)")
        print(f"Confidence: {result['probability']:.4f}")
        print(f"Probabilities: {result['probabilities']}")
    else:
        print(f"Error: {response.text}")


def test_xgboost_prediction():
    """Test XGBoost single prediction."""
    print("\nTesting XGBoost single prediction...")
    
    # Create sample data (14 features)
    sample_data = {
        "features": [0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.6, 0.4, 0.7, 0.3, 0.5, 0.2, 0.8, 0.4]
    }
    
    response = requests.post(
        "http://localhost:8000/predict/xgboost",
        json=sample_data
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']} (0=Healthy, 1=Unhealthy)")
        print(f"Confidence: {result['probability']:.4f}")
        print(f"Probabilities: {result['probabilities']}")
    else:
        print(f"Error: {response.text}")


def test_batch_prediction():
    """Test batch predictions for low-latency inference."""
    print("\nTesting XGBoost batch prediction...")
    
    # Create batch of 5 samples
    batch_data = {
        "samples": [
            [0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.6, 0.4, 0.7, 0.3, 0.5, 0.2, 0.8, 0.4],
            [0.2, 0.4, 0.3, 0.5, 0.1, 0.6, 0.2, 0.7, 0.3, 0.8, 0.4, 0.9, 0.5, 0.1],
            [1.5, 1.3, 1.8, 1.2, 1.9, 1.1, 1.6, 1.4, 1.7, 1.3, 1.5, 1.2, 1.8, 1.4],
            [0.6, 0.4, 0.9, 0.3, 1.0, 0.2, 0.7, 0.5, 0.8, 0.4, 0.6, 0.3, 0.9, 0.5],
            [1.2, 1.0, 1.5, 0.9, 1.6, 0.8, 1.3, 1.1, 1.4, 1.0, 1.2, 0.9, 1.5, 1.1]
        ]
    }
    
    response = requests.post(
        "http://localhost:8000/predict/xgboost/batch",
        json=batch_data
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Processed {len(result['predictions'])} samples:")
        for i, pred in enumerate(result['predictions']):
            print(f"  Sample {i+1}: Prediction={pred['prediction']}, Confidence={pred['probability']:.4f}")
    else:
        print(f"Error: {response.text}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("INFERENCE API TESTING")
    print("=" * 70)
    print("\nMake sure the API is running:")
    print("  cd api && python inference.py")
    print("  OR")
    print("  docker run -p 8000:8000 engine-health-mlops:latest")
    print("\n" + "=" * 70)
    
    try:
        # Test health
        health = test_health()
        
        if not health['models_loaded']['cnn'] and not health['models_loaded']['xgboost']:
            print("\n⚠️  Warning: No models are loaded!")
            print("Models need to be trained first and placed in the models/ directory.")
            print("Run: python run_pipeline.py")
        
        # Test CNN prediction
        if health['models_loaded']['cnn']:
            test_cnn_prediction()
        else:
            print("\n⚠️  Skipping CNN test - model not loaded")
        
        # Test XGBoost prediction
        if health['models_loaded']['xgboost']:
            test_xgboost_prediction()
            test_batch_prediction()
        else:
            print("\n⚠️  Skipping XGBoost test - model not loaded")
        
        print("\n" + "=" * 70)
        print("API TESTING COMPLETE!")
        print("=" * 70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the API is running on http://localhost:8000")
        print("\nStart the API with:")
        print("  uvicorn api.inference:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
