#!/usr/bin/env python3
"""
Comprehensive Error Handling and Retry Mechanisms
Production-grade error handling with automatic retries, circuit breakers,
fallback strategies, and detailed error tracking.
"""

import os
import sys
import json
import uuid
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps
import logging
import traceback
import inspect

# Retry and circuit breaker libraries
import tenacity
from tenacity import (
    retry, stop_after_attempt, wait_exponential, wait_fixed,
    retry_if_exception_type, retry_if_result, before_sleep_log
)

# FastAPI error handling
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import (
    http_exception_handler, request_validation_exception_handler
)
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# Monitoring and alerting
import structlog
from prometheus_client import Counter, Histogram, Gauge
import sentry_sdk
from sentry_sdk import capture_exception, capture_message, set_tag, set_extra

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Metrics
ERROR_COUNTER = Counter('errors_total', 'Total errors by type and severity', ['error_type', 'severity', 'component'])
RETRY_COUNTER = Counter('retries_total', 'Total retry attempts', ['operation', 'retry_reason'])
CIRCUIT_BREAKER_STATE = Gauge('circuit_breaker_state', 'Circuit breaker state (0=closed, 1=open, 2=half_open)', ['service'])
ERROR_RESOLUTION_TIME = Histogram('error_resolution_duration_seconds', 'Time to resolve errors', ['error_type'])
FALLBACK_USAGE = Counter('fallback_usage_total', 'Total fallback mechanism usage', ['fallback_type'])

# Setup logging
logger = structlog.get_logger(__name__)

# Configuration
class ErrorHandlingConfig:
    """Configuration for error handling system."""
    
    # Retry configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "1.0"))  # seconds
    RETRY_MAX_DELAY = float(os.getenv("RETRY_MAX_DELAY", "60.0"))  # seconds
    RETRY_MULTIPLIER = float(os.getenv("RETRY_MULTIPLIER", "2.0"))
    
    # Circuit breaker configuration
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60"))  # seconds
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION = int(os.getenv("CIRCUIT_BREAKER_EXPECTED_EXCEPTION", "10"))
    
    # Timeout configuration
    DEFAULT_TIMEOUT = float(os.getenv("DEFAULT_TIMEOUT", "30.0"))  # seconds
    PROCESSING_TIMEOUT = float(os.getenv("PROCESSING_TIMEOUT", "300.0"))  # seconds
    BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "3600.0"))  # seconds
    
    # Error tracking
    ENABLE_ERROR_TRACKING = bool(os.getenv("ENABLE_ERROR_TRACKING", "true").lower() == "true")
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    
    # Alerting
    ENABLE_ALERTING = bool(os.getenv("ENABLE_ALERTING", "false").lower() == "true")
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
    
    # Fallback configuration
    ENABLE_FALLBACKS = bool(os.getenv("ENABLE_FALLBACKS", "true").lower() == "true")
    FALLBACK_CACHE_TTL = int(os.getenv("FALLBACK_CACHE_TTL", "3600"))  # seconds

config = ErrorHandlingConfig()

# Enums
class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    VALIDATION = "validation"
    PROCESSING = "processing"
    INFRASTRUCTURE = "infrastructure"
    EXTERNAL_API = "external_api"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

class RetryStrategy(Enum):
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE = "immediate"
    NO_RETRY = "no_retry"

class CircuitBreakerState(Enum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2

# Custom Exceptions
class RecipeProcessingError(Exception):
    """Base exception for recipe processing errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, category: ErrorCategory = ErrorCategory.PROCESSING):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.severity = severity
        self.category = category
        self.timestamp = datetime.utcnow()
        self.trace_id = str(uuid.uuid4())

class ImageProcessingError(RecipeProcessingError):
    """Error during image processing."""
    pass

class FormatDetectionError(RecipeProcessingError):
    """Error during format detection."""
    pass

class OCRError(RecipeProcessingError):
    """Error during OCR processing."""
    pass

class IngredientParsingError(RecipeProcessingError):
    """Error during ingredient parsing."""
    pass

class ValidationError(RecipeProcessingError):
    """Error during data validation."""
    
    def __init__(self, message: str, validation_errors: List[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)
        self.validation_errors = validation_errors or []

class InfrastructureError(RecipeProcessingError):
    """Error related to infrastructure."""
    
    def __init__(self, message: str, service: str = None, **kwargs):
        super().__init__(message, category=ErrorCategory.INFRASTRUCTURE, 
                        severity=ErrorSeverity.HIGH, **kwargs)
        self.service = service

class ExternalAPIError(RecipeProcessingError):
    """Error when calling external APIs."""
    
    def __init__(self, message: str, api_name: str = None, status_code: int = None, **kwargs):
        super().__init__(message, category=ErrorCategory.EXTERNAL_API, **kwargs)
        self.api_name = api_name
        self.status_code = status_code

class ResourceError(RecipeProcessingError):
    """Error related to resource limitations."""
    
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        super().__init__(message, category=ErrorCategory.RESOURCE, 
                        severity=ErrorSeverity.HIGH, **kwargs)
        self.resource_type = resource_type

class TimeoutError(RecipeProcessingError):
    """Error due to operation timeout."""
    
    def __init__(self, message: str, operation: str = None, timeout_duration: float = None, **kwargs):
        super().__init__(message, category=ErrorCategory.TIMEOUT, **kwargs)
        self.operation = operation
        self.timeout_duration = timeout_duration

# Data Classes
@dataclass
class ErrorDetails:
    """Detailed error information."""
    error_id: str
    error_code: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: datetime
    trace_id: str
    component: str
    operation: str
    details: Dict[str, Any]
    stack_trace: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None

@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    timestamp: datetime
    error: Exception
    delay: float
    next_retry_at: Optional[datetime] = None

@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    state: CircuitBreakerState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    next_attempt_time: Optional[datetime]

# Circuit Breaker Implementation
class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, name: str, failure_threshold: int = None, recovery_timeout: int = None):
        self.name = name
        self.failure_threshold = failure_threshold or config.CIRCUIT_BREAKER_FAILURE_THRESHOLD
        self.recovery_timeout = recovery_timeout or config.CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.next_attempt_time = None
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        current_time = datetime.utcnow()
        
        # Check circuit breaker state
        if self.state == CircuitBreakerState.OPEN:
            if self.next_attempt_time and current_time < self.next_attempt_time:
                raise InfrastructureError(
                    f"Circuit breaker '{self.name}' is OPEN. Next attempt at {self.next_attempt_time}",
                    service=self.name
                )
            else:
                # Try to transition to half-open
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - update stats
            self.success_count += 1
            self.last_success_time = current_time
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                # Transition back to closed
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")
            
            CIRCUIT_BREAKER_STATE.labels(service=self.name).set(self.state.value)
            return result
            
        except Exception as e:
            # Failure - update stats
            self.failure_count += 1
            self.last_failure_time = current_time
            
            # Check if threshold exceeded
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.next_attempt_time = current_time + timedelta(seconds=self.recovery_timeout)
                logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN")
            
            CIRCUIT_BREAKER_STATE.labels(service=self.name).set(self.state.value)
            ERROR_COUNTER.labels(
                error_type=type(e).__name__,
                severity=getattr(e, 'severity', ErrorSeverity.MEDIUM).value,
                component=self.name
            ).inc()
            
            raise
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return CircuitBreakerStats(
            state=self.state,
            failure_count=self.failure_count,
            success_count=self.success_count,
            last_failure_time=self.last_failure_time,
            last_success_time=self.last_success_time,
            next_attempt_time=self.next_attempt_time
        )
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.next_attempt_time = None
        logger.info(f"Circuit breaker '{self.name}' manually reset")

# Retry Decorators
def retry_with_backoff(
    max_attempts: int = None,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay: float = None,
    max_delay: float = None,
    multiplier: float = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable = None
):
    """Decorator for retry with configurable backoff strategy."""
    
    max_attempts = max_attempts or config.MAX_RETRIES
    base_delay = base_delay or config.RETRY_BASE_DELAY
    max_delay = max_delay or config.RETRY_MAX_DELAY
    multiplier = multiplier or config.RETRY_MULTIPLIER
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except exceptions as e:
                    last_exception = e
                    
                    # Log retry attempt
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts + 1} failed for {func.__name__}",
                        error=str(e),
                        attempt=attempt + 1
                    )
                    
                    RETRY_COUNTER.labels(
                        operation=func.__name__,
                        retry_reason=type(e).__name__
                    ).inc()
                    
                    # Don't retry on last attempt
                    if attempt == max_attempts:
                        break
                    
                    # Calculate delay
                    if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                        delay = min(base_delay * (multiplier ** attempt), max_delay)
                    elif strategy == RetryStrategy.LINEAR_BACKOFF:
                        delay = min(base_delay * (attempt + 1), max_delay)
                    elif strategy == RetryStrategy.FIXED_DELAY:
                        delay = base_delay
                    else:  # IMMEDIATE
                        delay = 0
                    
                    # Custom retry callback
                    if on_retry:
                        await on_retry(attempt + 1, e, delay)
                    
                    # Wait before retry
                    if delay > 0:
                        await asyncio.sleep(delay)
            
            # All retries exhausted
            if last_exception:
                logger.error(f"All retry attempts exhausted for {func.__name__}", error=str(last_exception))
                raise last_exception
            
        return wrapper
    return decorator

# Error Context Manager
class ErrorContext:
    """Context manager for error handling and tracking."""
    
    def __init__(self, operation: str, component: str = "unknown", 
                 user_id: str = None, request_id: str = None):
        self.operation = operation
        self.component = component
        self.user_id = user_id
        self.request_id = request_id
        self.start_time = None
        self.error_details = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        logger.info(f"Starting operation: {self.operation}", component=self.component)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            # Success
            logger.info(
                f"Operation completed successfully: {self.operation}",
                component=self.component,
                duration=duration
            )
        else:
            # Error occurred
            self.error_details = self._create_error_details(exc_val, duration)
            await self._handle_error(self.error_details)
            
        return False  # Don't suppress exceptions
    
    def _create_error_details(self, exception: Exception, duration: float) -> ErrorDetails:
        """Create detailed error information."""
        if isinstance(exception, RecipeProcessingError):
            severity = exception.severity
            category = exception.category
            error_code = exception.error_code
            details = exception.details
        else:
            severity = ErrorSeverity.MEDIUM
            category = ErrorCategory.UNKNOWN
            error_code = type(exception).__name__
            details = {}
        
        return ErrorDetails(
            error_id=str(uuid.uuid4()),
            error_code=error_code,
            message=str(exception),
            severity=severity,
            category=category,
            timestamp=datetime.utcnow(),
            trace_id=getattr(exception, 'trace_id', str(uuid.uuid4())),
            component=self.component,
            operation=self.operation,
            details={
                **details,
                'duration': duration,
                'exception_type': type(exception).__name__
            },
            stack_trace=traceback.format_exc(),
            user_id=self.user_id,
            request_id=self.request_id
        )
    
    async def _handle_error(self, error_details: ErrorDetails):
        """Handle error with logging, metrics, and alerting."""
        # Update metrics
        ERROR_COUNTER.labels(
            error_type=error_details.error_code,
            severity=error_details.severity.value,
            component=error_details.component
        ).inc()
        
        # Log error
        logger.error(
            f"Error in {error_details.operation}",
            error_id=error_details.error_id,
            error_code=error_details.error_code,
            severity=error_details.severity.value,
            category=error_details.category.value,
            details=error_details.details
        )
        
        # Send to Sentry if enabled
        if config.ENABLE_ERROR_TRACKING and config.SENTRY_DSN:
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("component", error_details.component)
                scope.set_tag("operation", error_details.operation)
                scope.set_tag("severity", error_details.severity.value)
                scope.set_tag("category", error_details.category.value)
                scope.set_extra("error_details", asdict(error_details))
                
                if error_details.user_id:
                    scope.set_user({"id": error_details.user_id})
                
                capture_exception()
        
        # Send alerts for high severity errors
        if error_details.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            await self._send_alert(error_details)
    
    async def _send_alert(self, error_details: ErrorDetails):
        """Send alert for critical errors."""
        if not config.ENABLE_ALERTING:
            return
        
        try:
            # Implement your alerting mechanism here
            # For example, Slack webhook, PagerDuty, email, etc.
            logger.warning(f"Alert: {error_details.severity.value} error in {error_details.component}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

# Fallback Strategies
class FallbackManager:
    """Manages fallback strategies for failed operations."""
    
    def __init__(self):
        self.fallback_strategies = {}
    
    def register_fallback(self, operation: str, fallback_func: Callable):
        """Register a fallback function for an operation."""
        self.fallback_strategies[operation] = fallback_func
    
    async def execute_with_fallback(self, operation: str, primary_func: Callable, 
                                  *args, **kwargs) -> Any:
        """Execute function with fallback strategy."""
        try:
            # Try primary function
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            else:
                return primary_func(*args, **kwargs)
                
        except Exception as e:
            logger.warning(f"Primary function failed for {operation}, trying fallback", error=str(e))
            
            # Try fallback
            if operation in self.fallback_strategies:
                try:
                    fallback_func = self.fallback_strategies[operation]
                    FALLBACK_USAGE.labels(fallback_type=operation).inc()
                    
                    if asyncio.iscoroutinefunction(fallback_func):
                        return await fallback_func(*args, **kwargs)
                    else:
                        return fallback_func(*args, **kwargs)
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {operation}", error=str(fallback_error))
                    raise e  # Raise original error
            else:
                logger.error(f"No fallback registered for {operation}")
                raise

# FastAPI Error Handlers
class APIErrorHandler:
    """Centralized API error handling."""
    
    @staticmethod
    async def recipe_processing_error_handler(request: Request, exc: RecipeProcessingError) -> JSONResponse:
        """Handle recipe processing errors."""
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "severity": exc.severity.value,
                    "category": exc.category.value,
                    "trace_id": exc.trace_id,
                    "timestamp": exc.timestamp.isoformat(),
                    "details": exc.details
                }
            }
        )
    
    @staticmethod
    async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
        """Handle validation errors."""
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Validation failed",
                    "details": {
                        "validation_errors": exc.validation_errors if hasattr(exc, 'validation_errors') else [],
                        "field_errors": exc.errors() if hasattr(exc, 'errors') else []
                    }
                }
            }
        )
    
    @staticmethod
    async def infrastructure_error_handler(request: Request, exc: InfrastructureError) -> JSONResponse:
        """Handle infrastructure errors."""
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": "Service temporarily unavailable",
                    "service": exc.service,
                    "trace_id": exc.trace_id,
                    "retry_after": 60  # seconds
                }
            }
        )
    
    @staticmethod
    async def timeout_error_handler(request: Request, exc: TimeoutError) -> JSONResponse:
        """Handle timeout errors."""
        return JSONResponse(
            status_code=408,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": "Request timeout",
                    "operation": exc.operation,
                    "timeout_duration": exc.timeout_duration,
                    "trace_id": exc.trace_id
                }
            }
        )
    
    @staticmethod
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        error_id = str(uuid.uuid4())
        
        logger.error(
            f"Unhandled exception: {type(exc).__name__}",
            error_id=error_id,
            error=str(exc),
            path=request.url.path,
            method=request.method
        )
        
        # Send to Sentry
        if config.ENABLE_ERROR_TRACKING:
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("error_id", error_id)
                scope.set_extra("request_path", request.url.path)
                scope.set_extra("request_method", request.method)
                capture_exception()
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                    "error_id": error_id
                }
            }
        )

# Timeout Manager
class TimeoutManager:
    """Manages operation timeouts."""
    
    @staticmethod
    async def with_timeout(operation: Callable, timeout: float = None, 
                          operation_name: str = None) -> Any:
        """Execute operation with timeout."""
        timeout = timeout or config.DEFAULT_TIMEOUT
        operation_name = operation_name or getattr(operation, '__name__', 'unknown')
        
        try:
            if asyncio.iscoroutinefunction(operation):
                return await asyncio.wait_for(operation(), timeout=timeout)
            else:
                # For sync functions, run in executor
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(None, operation), 
                    timeout=timeout
                )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Operation '{operation_name}' timed out after {timeout} seconds",
                operation=operation_name,
                timeout_duration=timeout
            )

# Global instances
fallback_manager = FallbackManager()
error_handler = APIErrorHandler()
timeout_manager = TimeoutManager()

# Common circuit breakers
image_processing_circuit_breaker = CircuitBreaker("image_processing")
ocr_circuit_breaker = CircuitBreaker("ocr_processing")
database_circuit_breaker = CircuitBreaker("database")
external_api_circuit_breaker = CircuitBreaker("external_api")

# Utility functions
def handle_known_errors(func: Callable) -> Callable:
    """Decorator to convert known exceptions to custom errors."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise ValidationError(f"File not found: {e}")
        except PermissionError as e:
            raise InfrastructureError(f"Permission denied: {e}")
        except ConnectionError as e:
            raise ExternalAPIError(f"Connection failed: {e}")
        except ValueError as e:
            raise ValidationError(f"Invalid value: {e}")
        except MemoryError as e:
            raise ResourceError(f"Out of memory: {e}", resource_type="memory")
        except Exception as e:
            # Log unknown errors for investigation
            logger.warning(f"Unknown error type in {func.__name__}: {type(e).__name__}", error=str(e))
            raise
    
    return wrapper

def create_error_context(operation: str, **kwargs) -> ErrorContext:
    """Create error context for operation tracking."""
    return ErrorContext(operation, **kwargs)

# Health check for error handling system
async def health_check() -> Dict[str, Any]:
    """Check health of error handling system."""
    return {
        "error_tracking_enabled": config.ENABLE_ERROR_TRACKING,
        "alerting_enabled": config.ENABLE_ALERTING,
        "fallbacks_enabled": config.ENABLE_FALLBACKS,
        "circuit_breakers": {
            "image_processing": image_processing_circuit_breaker.get_stats(),
            "ocr_processing": ocr_circuit_breaker.get_stats(),
            "database": database_circuit_breaker.get_stats(),
            "external_api": external_api_circuit_breaker.get_stats()
        }
    }

if __name__ == "__main__":
    # Test error handling
    async def test_error_handling():
        # Test retry decorator
        @retry_with_backoff(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL_BACKOFF)
        async def failing_function():
            raise ValueError("Test error")
        
        # Test circuit breaker
        @image_processing_circuit_breaker
        async def test_function():
            raise Exception("Test error")
        
        # Test error context
        async with create_error_context("test_operation", component="test"):
            raise ValidationError("Test validation error")
    
    # Run test
    asyncio.run(test_error_handling())