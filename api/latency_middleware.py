"""Latency tracking middleware for FastAPI"""

import time
from fastapi import Request
from prometheus_client import Histogram

# Prometheus metrics
inference_latency = Histogram(
    'inference_latency_seconds',
    'Inference latency',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5)
)

async def latency_middleware(request: Request, call_next):
    """Measure request latency"""
    start_time = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start_time
    
    # Log to Prometheus
    inference_latency.observe(latency)
    
    # Add header
    response.headers["X-Latency-Ms"] = f"{latency * 1000:.2f}"
    
    return response
