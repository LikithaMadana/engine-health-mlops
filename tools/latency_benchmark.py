#!/usr/bin/env python3
"""Latency Measurement for Real-Time Automotive Applications"""

import time
import numpy as np
import torch
import statistics
from typing import Dict, Tuple

class LatencyBenchmark:
    def __init__(self, model, target_latency_ms: float = 100.0):
        self.model = model
        self.target_latency_ms = target_latency_ms
        
    def measure_single_inference(self, input_data: torch.Tensor) -> float:
        start = time.perf_counter()
        with torch.no_grad():
            _ = self.model(input_data)
        end = time.perf_counter()
        return (end - start) * 1000
    
    def benchmark(self, num_samples: int = 1000) -> Dict:
        latencies = []
        input_shape = (1, 6, 10)
        
        # Warmup
        for _ in range(10):
            test_input = torch.randn(input_shape)
            self.measure_single_inference(test_input)
        
        # Measure
        for _ in range(num_samples):
            test_input = torch.randn(input_shape)
            latency = self.measure_single_inference(test_input)
            latencies.append(latency)
        
        return {
            'mean_ms': statistics.mean(latencies),
            'median_ms': statistics.median(latencies),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'target_ms': self.target_latency_ms,
            'meets_target': np.percentile(latencies, 99) < self.target_latency_ms
        }


class LatencyOptimizer:
    @staticmethod
    def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
        model.eval()
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    
    @staticmethod
    def to_onnx(model: torch.nn.Module, output_path: str):
        dummy_input = torch.randn(1, 6, 10)
        torch.onnx.export(model, dummy_input, output_path,
                         export_params=True, opset_version=11)


if __name__ == "__main__":
    from src.models.cnn_model import EngineCNN
    
    model = EngineCNN(input_channels=6, window_size=10, hidden_channels=16)
    model.eval()
    
    benchmark = LatencyBenchmark(model, target_latency_ms=100.0)
    results = benchmark.benchmark(num_samples=1000)
    
    print("=== Latency Benchmark ===")
    print(f"Mean: {results['mean_ms']:.2f} ms")
    print(f"P95: {results['p95_ms']:.2f} ms")
    print(f"P99: {results['p99_ms']:.2f} ms")
    print(f"Target: {results['target_ms']:.2f} ms")
    print(f"Meets Target: {results['meets_target']}")
