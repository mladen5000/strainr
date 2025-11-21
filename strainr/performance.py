"""
Performance optimization utilities for production environments.

Provides caching, memory management, and batch processing utilities.
"""

import functools
import gc
import logging
import sys
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor and manage memory usage during processing."""

    def __init__(self, max_memory_mb: Optional[float] = None):
        """
        Initialize memory monitor.

        Args:
            max_memory_mb: Maximum memory threshold in MB. If exceeded, trigger GC.
        """
        self.max_memory_mb = max_memory_mb
        self.peak_memory_mb = 0.0
        self.gc_count = 0

    def get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            return mem_mb
        except ImportError:
            # Fallback if psutil not available
            return 0.0

    def check_and_collect(self) -> bool:
        """
        Check memory usage and trigger GC if needed.

        Returns:
            True if garbage collection was triggered
        """
        current_mb = self.get_current_memory_mb()
        self.peak_memory_mb = max(self.peak_memory_mb, current_mb)

        if self.max_memory_mb and current_mb > self.max_memory_mb:
            logger.warning(
                f"Memory usage ({current_mb:.1f} MB) exceeds threshold "
                f"({self.max_memory_mb:.1f} MB). Triggering garbage collection."
            )
            gc.collect()
            self.gc_count += 1
            new_mb = self.get_current_memory_mb()
            logger.info(f"After GC: {new_mb:.1f} MB (freed {current_mb - new_mb:.1f} MB)")
            return True

        return False

    def get_stats(self) -> dict:
        """Get memory monitoring statistics."""
        return {
            'current_memory_mb': self.get_current_memory_mb(),
            'peak_memory_mb': self.peak_memory_mb,
            'gc_triggered_count': self.gc_count
        }


class BatchProcessor:
    """Process large datasets in batches for memory efficiency."""

    def __init__(self, batch_size: int = 1000, enable_progress: bool = True):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items per batch
            enable_progress: Show progress bars if available
        """
        self.batch_size = batch_size
        self.enable_progress = enable_progress
        self.total_processed = 0

    def process_batches(
        self,
        items: list,
        process_fn: Callable[[list], Any],
        description: str = "Processing"
    ) -> list:
        """
        Process items in batches.

        Args:
            items: List of items to process
            process_fn: Function to apply to each batch
            description: Description for progress bar

        Returns:
            List of results from processing each batch
        """
        results = []
        num_batches = (len(items) + self.batch_size - 1) // self.batch_size

        try:
            from tqdm import tqdm
            iterator = tqdm(
                range(num_batches),
                desc=description,
                disable=not self.enable_progress
            )
        except ImportError:
            iterator = range(num_batches)

        for batch_idx in iterator:
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(items))
            batch = items[start_idx:end_idx]

            result = process_fn(batch)
            results.append(result)
            self.total_processed += len(batch)

        return results


def lru_cache_with_size(maxsize: int = 128):
    """
    LRU cache decorator with size tracking.

    Like functools.lru_cache but logs cache statistics.
    """
    def decorator(func: Callable) -> Callable:
        cached_func = functools.lru_cache(maxsize=maxsize)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = cached_func(*args, **kwargs)
            info = cached_func.cache_info()

            # Log cache stats every 100 calls
            if (info.hits + info.misses) % 100 == 0:
                hit_rate = info.hits / (info.hits + info.misses) if info.hits + info.misses > 0 else 0
                logger.debug(
                    f"Cache stats for {func.__name__}: "
                    f"hits={info.hits}, misses={info.misses}, "
                    f"hit_rate={hit_rate:.2%}, size={info.currsize}/{maxsize}"
                )

            return result

        wrapper.cache_info = cached_func.cache_info
        wrapper.cache_clear = cached_func.cache_clear
        return wrapper

    return decorator


class ChunkedFileReader:
    """Read large files in chunks to avoid loading entire file into memory."""

    def __init__(self, chunk_size_mb: float = 100):
        """
        Initialize chunked file reader.

        Args:
            chunk_size_mb: Size of each chunk in megabytes
        """
        self.chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)

    def read_chunks(self, filepath: str):
        """
        Generator that yields file chunks.

        Args:
            filepath: Path to file to read

        Yields:
            Chunks of file data as bytes
        """
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size_bytes)
                if not chunk:
                    break
                yield chunk
