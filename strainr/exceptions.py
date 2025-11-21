"""
Custom exception hierarchy for StrainR.

Provides granular exception types for better error handling, debugging,
and user feedback in production environments.
"""


class StrainRException(Exception):
    """Base exception for all StrainR errors."""

    def __init__(self, message: str, details: dict = None):
        """
        Initialize exception with message and optional details.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# Database-related exceptions
class DatabaseException(StrainRException):
    """Base exception for database-related errors."""
    pass


class DatabaseNotFoundError(DatabaseException):
    """Database file not found."""
    pass


class DatabaseCorruptedError(DatabaseException):
    """Database file is corrupted or invalid format."""
    pass


class DatabaseLoadError(DatabaseException):
    """Error loading database into memory."""
    pass


class IncompatibleDatabaseError(DatabaseException):
    """Database version or format incompatible with current software."""
    pass


# Input validation exceptions
class ValidationException(StrainRException):
    """Base exception for validation errors."""
    pass


class InvalidInputFileError(ValidationException):
    """Input file is invalid or cannot be read."""
    pass


class InvalidParameterError(ValidationException):
    """Parameter value is invalid or out of range."""
    pass


class IncompatibleInputError(ValidationException):
    """Input files are incompatible with each other or database."""
    pass


# Processing exceptions
class ProcessingException(StrainRException):
    """Base exception for processing errors."""
    pass


class KmerExtractionError(ProcessingException):
    """Error during k-mer extraction."""
    pass


class ClassificationError(ProcessingException):
    """Error during read classification."""
    pass


class MemoryError(ProcessingException):
    """Insufficient memory for operation."""
    pass


class TimeoutError(ProcessingException):
    """Operation exceeded timeout limit."""
    pass


# Resource exceptions
class ResourceException(StrainRException):
    """Base exception for resource-related errors."""
    pass


class InsufficientResourcesError(ResourceException):
    """Insufficient system resources (disk, memory, etc)."""
    pass


class OutputWriteError(ResourceException):
    """Error writing output files."""
    pass


# Analysis exceptions
class AnalysisException(StrainRException):
    """Base exception for analysis errors."""
    pass


class InsufficientDataError(AnalysisException):
    """Insufficient data for analysis."""
    pass


class AmbiguousResultError(AnalysisException):
    """Result is ambiguous and cannot be resolved."""
    pass
