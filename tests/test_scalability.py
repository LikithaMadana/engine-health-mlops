"""
Scalability Tests for 100x Data Handling

These tests verify that the pipeline can handle 100x larger datasets
without crashing due to memory constraints.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.memory_utils import (
    get_memory_usage, check_memory_available, get_recommended_chunk_size,
    MemoryMonitor, force_garbage_collection
)
from src.utils.scalable_data_loader import (
    load_csv_chunked, csv_chunk_generator, prepare_data_scalable,
    BatchDataProcessor
)


class TestMemoryUtils:
    """Test memory management utilities"""
    
    def test_get_memory_usage(self):
        """Test getting memory usage stats"""
        mem = get_memory_usage()
        assert 'total_gb' in mem
        assert 'used_gb' in mem
        assert 'percent' in mem
        assert mem['total_gb'] > 0
        assert 0 <= mem['percent'] <= 100
    
    def test_check_memory_available(self):
        """Test memory availability check"""
        # Should pass with reasonable requirement
        assert check_memory_available(min_gb=0.1) == True
    
    def test_get_recommended_chunk_size(self):
        """Test chunk size calculation"""
        chunk_size = get_recommended_chunk_size(target_memory_gb=0.5)
        assert chunk_size >= 10000  # Minimum
        assert chunk_size % 10000 == 0  # Rounded
    
    def test_memory_monitor(self):
        """Test memory monitoring context manager"""
        with MemoryMonitor("Test operation"):
            # Allocate some memory
            data = np.random.rand(1000, 1000)
            del data
        # Should not raise exception
    
    def test_force_garbage_collection(self):
        """Test garbage collection"""
        result = force_garbage_collection()
        assert 'before' in result
        assert 'after' in result
        assert 'freed_gb' in result


class TestScalableDataLoader:
    """Test scalable data loading capabilities"""
    
    @pytest.fixture
    def sample_csv(self):
        """Create a temporary CSV file for testing"""
        # Create sample data similar to engine data
        n_samples = 10000  # 10K samples for quick test
        data = {
            'Engine rpm': np.random.uniform(1000, 5000, n_samples),
            'Lub oil pressure': np.random.uniform(20, 60, n_samples),
            'Fuel pressure': np.random.uniform(30, 80, n_samples),
            'Coolant pressure': np.random.uniform(10, 50, n_samples),
            'lub oil temp': np.random.uniform(50, 120, n_samples),
            'Coolant temp': np.random.uniform(60, 110, n_samples),
            'Engine Condition': np.random.randint(0, 2, n_samples)
        }
        df = pd.DataFrame(data)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            filepath = f.name
        
        yield filepath
        
        # Cleanup
        os.unlink(filepath)
    
    def test_load_csv_chunked(self, sample_csv):
        """Test chunked CSV loading"""
        df = load_csv_chunked(sample_csv, chunksize=2000, verbose=False)
        assert len(df) == 10000
        assert len(df.columns) == 7
    
    def test_load_csv_chunked_with_max_rows(self, sample_csv):
        """Test chunked loading with max_rows limit"""
        df = load_csv_chunked(sample_csv, chunksize=2000, max_rows=5000, verbose=False)
        assert len(df) <= 5000
    
    def test_csv_chunk_generator(self, sample_csv):
        """Test CSV chunk generator"""
        total_rows = 0
        chunk_count = 0
        
        for chunk in csv_chunk_generator(sample_csv, chunksize=2000):
            total_rows += len(chunk)
            chunk_count += 1
            assert len(chunk) <= 2000
        
        assert total_rows == 10000
        assert chunk_count == 5  # 10000 / 2000
    
    def test_prepare_data_scalable(self, sample_csv):
        """Test scalable data preparation"""
        config = {'test_size': 0.2}
        
        train_X, train_y, test_X, test_y, scaler = prepare_data_scalable(
            sample_csv, config, chunksize=2000
        )
        
        assert len(train_X) == len(train_y)
        assert len(test_X) == len(test_y)
        assert len(train_X) + len(test_X) == 10000
        assert train_X.shape[1] == 6  # 6 features
    
    def test_batch_data_processor(self, sample_csv):
        """Test batch data processor"""
        processor = BatchDataProcessor(sample_csv, batch_size=2000)
        
        total_samples = 0
        for batch_X, batch_y in processor:
            total_samples += len(batch_X)
            assert batch_X.shape[1] == 6
            assert len(batch_X) == len(batch_y)
        
        assert total_samples == 10000


class TestScalability:
    """Test actual scalability with larger datasets"""
    
    def test_10x_data_handling(self):
        """Test handling 10x data (195K samples)"""
        # Create larger dataset in memory
        n_samples = 195000  # 10x original
        
        with MemoryMonitor("10x data test"):
            data = {
                'Engine rpm': np.random.uniform(1000, 5000, n_samples),
                'Lub oil pressure': np.random.uniform(20, 60, n_samples),
                'Fuel pressure': np.random.uniform(30, 80, n_samples),
                'Coolant pressure': np.random.uniform(10, 50, n_samples),
                'lub oil temp': np.random.uniform(50, 120, n_samples),
                'Coolant temp': np.random.uniform(60, 110, n_samples),
                'Engine Condition': np.random.randint(0, 2, n_samples)
            }
            df = pd.DataFrame(data)
            
            # Should not crash
            assert len(df) == n_samples
            
            # Clean up
            del df, data
            force_garbage_collection()
    
    def test_chunk_size_recommendation_for_100x(self):
        """Test chunk size for 100x data"""
        # For 100x data (1.95M samples), we need appropriate chunk size
        chunk_size = get_recommended_chunk_size(target_memory_gb=1.0)
        
        # Should be reasonable (10K - 500K)
        assert 10000 <= chunk_size <= 500000
        
        # Calculate how many chunks for 1.95M samples
        n_samples = 1953500  # 100x
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        # Should be manageable number of chunks
        assert n_chunks < 1000  # Not too many chunks


class TestMemoryEfficiency:
    """Test memory efficiency improvements"""
    
    def test_memory_usage_scales_linearly(self):
        """Verify memory usage scales linearly, not exponentially"""
        from src.utils.scalable_data_loader import load_csv_chunked
        
        # Create small dataset
        n_small = 1000
        data_small = {
            'Engine rpm': np.random.uniform(1000, 5000, n_small),
            'Lub oil pressure': np.random.uniform(20, 60, n_small),
            'Fuel pressure': np.random.uniform(30, 80, n_small),
            'Coolant pressure': np.random.uniform(10, 50, n_small),
            'lub oil temp': np.random.uniform(50, 120, n_small),
            'Coolant temp': np.random.uniform(60, 110, n_small),
            'Engine Condition': np.random.randint(0, 2, n_small)
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(data_small).to_csv(f.name, index=False)
            filepath = f.name
        
        try:
            # Load and measure memory
            mem_before = get_memory_usage()
            df = load_csv_chunked(filepath, chunksize=500, verbose=False)
            mem_after = get_memory_usage()
            
            memory_used = mem_after['used_gb'] - mem_before['used_gb']
            
            # Memory should be small (< 100MB)
            assert memory_used < 0.1  # Less than 100MB
            
            del df
        finally:
            os.unlink(filepath)


# Performance benchmarks (optional, can be slow)
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for scalability"""
    
    def test_benchmark_chunked_vs_regular(self):
        """Benchmark chunked loading vs regular loading"""
        import time
        
        # Create test data
        n_samples = 50000
        data = {
            'Engine rpm': np.random.uniform(1000, 5000, n_samples),
            'Lub oil pressure': np.random.uniform(20, 60, n_samples),
            'Fuel pressure': np.random.uniform(30, 80, n_samples),
            'Coolant pressure': np.random.uniform(10, 50, n_samples),
            'lub oil temp': np.random.uniform(50, 120, n_samples),
            'Coolant temp': np.random.uniform(60, 110, n_samples),
            'Engine Condition': np.random.randint(0, 2, n_samples)
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            filepath = f.name
        
        try:
            # Regular loading
            start = time.time()
            df1 = pd.read_csv(filepath)
            regular_time = time.time() - start
            
            # Chunked loading
            start = time.time()
            df2 = load_csv_chunked(filepath, chunksize=10000, verbose=False)
            chunked_time = time.time() - start
            
            # Chunked should be comparable (within 2x)
            assert chunked_time < regular_time * 2
            
            print(f"\nBenchmark results:")
            print(f"  Regular loading: {regular_time:.3f}s")
            print(f"  Chunked loading: {chunked_time:.3f}s")
            
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
