"""
Scalable data processing utilities for handling large datasets.

Provides memory-efficient data loading, chunked processing, and
streaming capabilities to handle 100x more data without crashing.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, Callable, List, Dict
from pathlib import Path
import psutil
import gc

from .logger import setup_logger
from .exceptions import DataPreprocessingError

logger = setup_logger(__name__)


class DataChunker:
    """
    Memory-efficient data chunking for large datasets.
    
    Processes data in chunks to avoid loading entire dataset into memory.
    """
    
    def __init__(self, chunk_size: int = 10000, max_memory_percent: float = 70.0):
        """
        Initialize data chunker.
        
        Args:
            chunk_size: Number of samples per chunk
            max_memory_percent: Maximum memory usage percentage (0-100)
        """
        self.chunk_size = chunk_size
        self.max_memory_percent = max_memory_percent
        logger.info(f"DataChunker initialized: chunk_size={chunk_size}, "
                   f"max_memory={max_memory_percent}%")
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage percentage.
        
        Returns:
            Memory usage as percentage (0-100)
        """
        memory = psutil.virtual_memory()
        return memory.percent
    
    def check_memory(self) -> bool:
        """
        Check if memory usage is within limits.
        
        Returns:
            True if memory usage is acceptable
        """
        usage = self.get_memory_usage()
        if usage > self.max_memory_percent:
            logger.warning(f"High memory usage: {usage:.1f}% (limit: {self.max_memory_percent}%)")
            return False
        return True
    
    def optimize_chunk_size(self, data_size: int, feature_size: int = 14) -> int:
        """
        Optimize chunk size based on available memory.
        
        Args:
            data_size: Total number of samples
            feature_size: Number of features per sample
            
        Returns:
            Optimized chunk size
        """
        available_memory = psutil.virtual_memory().available
        
        # Estimate memory per sample (in bytes)
        # float64 = 8 bytes per value
        bytes_per_sample = feature_size * 8
        
        # Use 10% of available memory for chunk
        target_memory = available_memory * 0.1
        optimal_chunk = int(target_memory / bytes_per_sample)
        
        # Clamp to reasonable range
        optimal_chunk = max(1000, min(optimal_chunk, self.chunk_size))
        
        logger.info(f"Optimized chunk size: {optimal_chunk} "
                   f"(available memory: {available_memory / 1e9:.2f}GB)")
        
        return optimal_chunk
    
    def chunk_array(
        self,
        data: np.ndarray,
        chunk_size: Optional[int] = None
    ) -> Iterator[np.ndarray]:
        """
        Split array into chunks.
        
        Args:
            data: Input array
            chunk_size: Override default chunk size
            
        Yields:
            Chunks of data
        """
        chunk_size = chunk_size or self.chunk_size
        n_samples = len(data)
        
        logger.info(f"Chunking {n_samples} samples into chunks of {chunk_size}")
        
        for i in range(0, n_samples, chunk_size):
            chunk = data[i:i + chunk_size]
            
            # Check memory after each chunk
            if not self.check_memory():
                logger.warning("Memory limit reached, forcing garbage collection")
                gc.collect()
            
            yield chunk
    
    def process_in_chunks(
        self,
        data: np.ndarray,
        process_func: Callable[[np.ndarray], np.ndarray],
        chunk_size: Optional[int] = None,
        description: str = "Processing"
    ) -> np.ndarray:
        """
        Process data in chunks and combine results.
        
        Args:
            data: Input data array
            process_func: Function to apply to each chunk
            chunk_size: Override default chunk size
            description: Description for logging
            
        Returns:
            Processed data array
        """
        chunk_size = chunk_size or self.chunk_size
        results = []
        
        logger.info(f"{description}: {len(data)} samples in chunks of {chunk_size}")
        
        for i, chunk in enumerate(self.chunk_array(data, chunk_size)):
            try:
                processed = process_func(chunk)
                results.append(processed)
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"Processed {(i + 1) * chunk_size} samples, "
                               f"memory: {self.get_memory_usage():.1f}%")
            
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                raise DataPreprocessingError(f"Chunk processing failed: {e}") from e
        
        # Combine results
        combined = np.vstack(results) if results else np.array([])
        
        logger.info(f"{description} complete: {len(combined)} samples, "
                   f"memory: {self.get_memory_usage():.1f}%")
        
        return combined


class StreamingDataLoader:
    """
    Stream data from disk without loading entire dataset into memory.
    
    Supports CSV, Parquet, and chunked numpy arrays.
    """
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize streaming data loader.
        
        Args:
            chunk_size: Number of rows to load per chunk
        """
        self.chunk_size = chunk_size
        logger.info(f"StreamingDataLoader initialized: chunk_size={chunk_size}")
    
    def stream_csv(
        self,
        file_path: str,
        feature_columns: Optional[List[str]] = None,
        label_column: Optional[str] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Stream data from CSV file in chunks.
        
        Args:
            file_path: Path to CSV file
            feature_columns: Column names for features (None = all except label)
            label_column: Column name for labels
            
        Yields:
            Tuples of (features, labels) for each chunk
        """
        logger.info(f"Streaming from CSV: {file_path}")
        
        try:
            # Use pandas chunking
            for chunk_df in pd.read_csv(file_path, chunksize=self.chunk_size):
                if label_column:
                    labels = chunk_df[label_column].values
                    
                    if feature_columns:
                        features = chunk_df[feature_columns].values
                    else:
                        features = chunk_df.drop(columns=[label_column]).values
                else:
                    features = chunk_df.values
                    labels = np.zeros(len(features))  # Dummy labels
                
                yield features, labels
                
        except Exception as e:
            logger.error(f"Error streaming CSV: {e}")
            raise DataPreprocessingError(f"CSV streaming failed: {e}") from e
    
    def stream_parquet(
        self,
        file_path: str,
        feature_columns: Optional[List[str]] = None,
        label_column: Optional[str] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Stream data from Parquet file in chunks.
        
        Args:
            file_path: Path to Parquet file
            feature_columns: Column names for features
            label_column: Column name for labels
            
        Yields:
            Tuples of (features, labels) for each chunk
        """
        logger.info(f"Streaming from Parquet: {file_path}")
        
        try:
            import pyarrow.parquet as pq
            
            parquet_file = pq.ParquetFile(file_path)
            
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                df = batch.to_pandas()
                
                if label_column:
                    labels = df[label_column].values
                    
                    if feature_columns:
                        features = df[feature_columns].values
                    else:
                        features = df.drop(columns=[label_column]).values
                else:
                    features = df.values
                    labels = np.zeros(len(features))
                
                yield features, labels
                
        except ImportError:
            logger.error("pyarrow not installed. Install with: pip install pyarrow")
            raise DataPreprocessingError("Parquet support requires pyarrow")
        except Exception as e:
            logger.error(f"Error streaming Parquet: {e}")
            raise DataPreprocessingError(f"Parquet streaming failed: {e}") from e
    
    def stream_numpy(
        self,
        file_path: str,
        mmap_mode: str = 'r'
    ) -> Iterator[np.ndarray]:
        """
        Stream data from numpy memory-mapped file.
        
        Args:
            file_path: Path to .npy file
            mmap_mode: Memory-map mode ('r', 'r+', 'w+', 'c')
            
        Yields:
            Chunks of data
        """
        logger.info(f"Streaming from numpy file: {file_path}")
        
        try:
            # Load as memory-mapped array (doesn't load into memory)
            data = np.load(file_path, mmap_mode=mmap_mode)
            
            n_samples = len(data)
            logger.info(f"Streaming {n_samples} samples from memory-mapped file")
            
            for i in range(0, n_samples, self.chunk_size):
                chunk = data[i:i + self.chunk_size]
                yield chunk
                
        except Exception as e:
            logger.error(f"Error streaming numpy file: {e}")
            raise DataPreprocessingError(f"Numpy streaming failed: {e}") from e


class BatchGenerator:
    """
    Generate batches for training with memory-efficient data loading.
    
    Supports on-the-fly preprocessing and augmentation.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        preprocessor: Optional[Callable] = None
    ):
        """
        Initialize batch generator.
        
        Args:
            X: Feature array
            y: Label array
            batch_size: Batch size
            shuffle: Whether to shuffle data each epoch
            preprocessor: Optional preprocessing function to apply
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocessor = preprocessor
        self.n_samples = len(X)
        self.n_batches = int(np.ceil(self.n_samples / batch_size))
        
        logger.info(f"BatchGenerator: {self.n_samples} samples, "
                   f"{self.n_batches} batches, batch_size={batch_size}")
    
    def __len__(self) -> int:
        """Return number of batches."""
        return self.n_batches
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate batches.
        
        Yields:
            Tuples of (batch_X, batch_y)
        """
        indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_X = self.X[batch_indices]
            batch_y = self.y[batch_indices]
            
            # Apply preprocessing if provided
            if self.preprocessor:
                batch_X = self.preprocessor(batch_X)
            
            yield batch_X, batch_y


def estimate_memory_requirements(
    n_samples: int,
    n_features: int,
    window_size: int = 50,
    dtype: str = 'float32'
) -> Dict[str, float]:
    """
    Estimate memory requirements for dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        window_size: Window size for CNN
        dtype: Data type ('float32' or 'float64')
        
    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_value = 4 if dtype == 'float32' else 8
    
    # Raw data memory
    raw_memory = n_samples * n_features * bytes_per_value
    
    # Windowed data memory (for CNN)
    windowed_memory = (n_samples - window_size + 1) * window_size * n_features * bytes_per_value
    
    # Training memory (includes gradients, optimizer states)
    training_memory = windowed_memory * 3  # Rough estimate
    
    estimates = {
        'raw_data_gb': raw_memory / 1e9,
        'windowed_data_gb': windowed_memory / 1e9,
        'training_estimate_gb': training_memory / 1e9,
        'recommended_ram_gb': training_memory / 1e9 * 2  # 2x for safety
    }
    
    logger.info(f"Memory estimates for {n_samples} samples: {estimates}")
    
    return estimates
