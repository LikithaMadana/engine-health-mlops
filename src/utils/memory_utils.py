"""
Memory Management Utilities for Large-Scale ML Pipelines

This module provides memory monitoring, management, and optimization
utilities to handle 100x larger datasets without crashing.
"""

import psutil
import gc
import sys
import logging
from typing import Optional, Callable
import torch

logger = logging.getLogger(__name__)


def get_memory_usage():
    """
    Get current memory usage statistics.
    
    Returns:
        dict: Memory statistics including used, available, and percent
    """
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024**3),
        'available_gb': mem.available / (1024**3),
        'used_gb': mem.used / (1024**3),
        'percent': mem.percent
    }


def check_memory_available(min_gb: float = 2.0) -> bool:
    """
    Check if minimum memory is available.
    
    Args:
        min_gb: Minimum required memory in GB
        
    Returns:
        bool: True if enough memory available
    """
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    
    if available_gb < min_gb:
        logger.warning(f"Low memory! Available: {available_gb:.2f}GB, Required: {min_gb:.2f}GB")
        return False
    return True


def memory_guard(max_percent: float = 90.0):
    """
    Decorator to guard functions against running out of memory.
    
    Args:
        max_percent: Maximum allowed memory usage percentage
        
    Raises:
        MemoryError: If memory usage exceeds threshold
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            mem = psutil.virtual_memory()
            if mem.percent > max_percent:
                raise MemoryError(
                    f"Memory usage too high: {mem.percent:.1f}% > {max_percent:.1f}%"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def force_garbage_collection():
    """
    Force garbage collection to free memory.
    
    Returns:
        dict: Memory stats before and after GC
    """
    before = get_memory_usage()
    
    # Collect garbage
    gc.collect()
    
    # If PyTorch is available, clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    after = get_memory_usage()
    
    freed_gb = before['used_gb'] - after['used_gb']
    logger.info(f"GC freed {freed_gb:.2f}GB of memory")
    
    return {'before': before, 'after': after, 'freed_gb': freed_gb}


def get_recommended_chunk_size(target_memory_gb: float = 1.0, 
                               row_size_bytes: int = 100) -> int:
    """
    Calculate recommended chunk size for processing large files.
    
    Args:
        target_memory_gb: Target memory usage per chunk in GB
        row_size_bytes: Estimated bytes per row
        
    Returns:
        int: Recommended number of rows per chunk
    """
    target_bytes = target_memory_gb * (1024**3)
    chunk_size = int(target_bytes / row_size_bytes)
    
    # Round to nearest 10,000
    chunk_size = (chunk_size // 10000) * 10000
    
    return max(10000, chunk_size)  # Minimum 10K rows


class MemoryMonitor:
    """
    Context manager for monitoring memory usage during operations.
    
    Example:
        >>> with MemoryMonitor("Data loading"):
        ...     df = pd.read_csv('large_file.csv')
    """
    
    def __init__(self, operation_name: str = "Operation", 
                 max_percent: float = 95.0):
        self.operation_name = operation_name
        self.max_percent = max_percent
        self.start_mem = None
        
    def __enter__(self):
        self.start_mem = get_memory_usage()
        logger.info(
            f"[{self.operation_name}] Started - Memory: "
            f"{self.start_mem['used_gb']:.2f}GB / "
            f"{self.start_mem['total_gb']:.2f}GB "
            f"({self.start_mem['percent']:.1f}%)"
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_mem = get_memory_usage()
        delta_gb = end_mem['used_gb'] - self.start_mem['used_gb']
        
        logger.info(
            f"[{self.operation_name}] Completed - Memory: "
            f"{end_mem['used_gb']:.2f}GB ({end_mem['percent']:.1f}%) - "
            f"Delta: {delta_gb:+.2f}GB"
        )
        
        # Check if we exceeded threshold
        if end_mem['percent'] > self.max_percent:
            logger.warning(
                f"Memory usage high after {self.operation_name}: "
                f"{end_mem['percent']:.1f}% > {self.max_percent:.1f}%"
            )
            # Trigger GC
            force_garbage_collection()


def optimize_pandas_dtypes(df, inplace=True):
    """
    Optimize pandas DataFrame dtypes to reduce memory usage.
    
    Args:
        df: pandas DataFrame
        inplace: Whether to modify DataFrame in place
        
    Returns:
        DataFrame with optimized dtypes
    """
    import pandas as pd
    
    if not inplace:
        df = df.copy()
    
    # Optimize integers
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Optimize floats
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Memory Management Utils Demo ===\n")
    
    # 1. Check current memory
    mem = get_memory_usage()
    print(f"Current Memory Usage:")
    print(f"  Total: {mem['total_gb']:.2f} GB")
    print(f"  Used: {mem['used_gb']:.2f} GB")
    print(f"  Available: {mem['available_gb']:.2f} GB")
    print(f"  Percent: {mem['percent']:.1f}%\n")
    
    # 2. Get recommended chunk size
    chunk_size = get_recommended_chunk_size(target_memory_gb=0.5)
    print(f"Recommended chunk size: {chunk_size:,} rows\n")
    
    # 3. Test memory monitor
    with MemoryMonitor("Test operation"):
        # Simulate some memory allocation
        import numpy as np
        large_array = np.random.rand(1000, 1000)
        del large_array
    
    print("\n=== Demo Complete ===")
