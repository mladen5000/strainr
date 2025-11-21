"""
Production-grade logging configuration for StrainR.

Provides structured logging with JSON formatting, log rotation,
and performance tracking capabilities.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs logs as JSON for easy parsing by log aggregation systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class PerformanceLogger:
    """
    Logger for performance metrics and monitoring.

    Tracks operation times, throughput, and resource usage.
    """

    def __init__(self, logger_name: str = 'strainr.performance'):
        """Initialize performance logger."""
        self.logger = logging.getLogger(logger_name)
        self.metrics: Dict[str, list] = {}

    def log_operation_time(self, operation: str, duration_seconds: float, **kwargs):
        """
        Log operation timing.

        Args:
            operation: Name of the operation
            duration_seconds: How long it took
            **kwargs: Additional context (e.g., items_processed, rate)
        """
        if operation not in self.metrics:
            self.metrics[operation] = []

        self.metrics[operation].append(duration_seconds)

        extra = {
            'extra_fields': {
                'operation': operation,
                'duration_seconds': duration_seconds,
                **kwargs
            }
        }

        self.logger.info(
            f"{operation} completed in {duration_seconds:.2f}s",
            extra=extra
        )

    def log_throughput(self, operation: str, items: int, duration_seconds: float):
        """
        Log throughput metrics.

        Args:
            operation: Name of the operation
            items: Number of items processed
            duration_seconds: Time taken
        """
        rate = items / duration_seconds if duration_seconds > 0 else 0
        self.log_operation_time(
            operation,
            duration_seconds,
            items_processed=items,
            items_per_second=rate
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all tracked operations."""
        import numpy as np

        summary = {}
        for operation, durations in self.metrics.items():
            summary[operation] = {
                'count': len(durations),
                'total_seconds': sum(durations),
                'mean_seconds': np.mean(durations),
                'median_seconds': np.median(durations),
                'min_seconds': min(durations),
                'max_seconds': max(durations)
            }

        return summary


def setup_production_logging(
    log_dir: Optional[Path] = None,
    log_level: str = 'INFO',
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configure production-grade logging for StrainR.

    Args:
        log_dir: Directory for log files (defaults to ./logs)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: Enable console (stdout) logging
        enable_file: Enable file logging
        enable_json: Use JSON formatting for structured logs
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup log files to keep

    Returns:
        Configured root logger

    Example:
        >>> logger = setup_production_logging(
        ...     log_dir=Path('logs'),
        ...     log_level='INFO',
        ...     enable_json=True
        ... )
        >>> logger.info("Application started")
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Choose formatter
    if enable_json:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if enable_file:
        log_file = log_dir / f'strainr_{datetime.now():%Y%m%d}.log'
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Separate error log
        error_log = log_dir / f'strainr_errors_{datetime.now():%Y%m%d}.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_log,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)

    return root_logger


class LogContext:
    """Context manager for adding contextual information to logs."""

    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize log context.

        Args:
            logger: Logger to add context to
            **context: Key-value pairs to add to all log messages
        """
        self.logger = logger
        self.context = context
        self.old_factory = None

    def __enter__(self):
        """Enter context and modify log record factory."""
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            if not hasattr(record, 'extra_fields'):
                record.extra_fields = {}
            record.extra_fields.update(self.context)
            return record

        logging.setLogRecordFactory(record_factory)
        self.old_factory = old_factory
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original factory."""
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)
        return False
