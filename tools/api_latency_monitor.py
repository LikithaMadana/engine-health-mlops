#!/usr/bin/env python3
"""Real-time API latency monitoring"""

import time
import requests
import statistics
from typing import Dict

def measure_api_latency(url: str, payload: Dict, num_requests: int = 100) -> Dict:
    latencies = []
    
    for _ in range(num_requests):
        start = time.perf_counter()
        response = requests.post(url, json=payload, timeout=5)
        end = time.perf_counter()
        
        if response.status_code == 200:
            latencies.append((end - start) * 1000)
    
    import numpy as np
    return {
        'mean_ms': statistics.mean(latencies),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99)
    }


if __name__ == "__main__":
    url = "http://localhost:8000/predict"
    payload = {
        "sensor_1": 1500.0,
        "sensor_2": 45.0,
        "sensor_3": 350.0,
        "sensor_4": 12.0,
        "sensor_5": 85.0,
        "sensor_6": 75.0
    }
    
    print("Testing API latency...")
    results = measure_api_latency(url, payload, num_requests=100)
    print(f"Mean: {results['mean_ms']:.2f} ms")
    print(f"P95: {results['p95_ms']:.2f} ms")
    print(f"P99: {results['p99_ms']:.2f} ms")
