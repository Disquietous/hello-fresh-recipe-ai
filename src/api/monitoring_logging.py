#!/usr/bin/env python3
"""
Comprehensive Monitoring and Logging System
Production-grade monitoring with metrics, tracing, structured logging,
health checks, and observability for recipe processing system.
"""

import os
import sys
import json
import uuid
import time
import asyncio
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps
import logging
import socket
import threading
from contextlib import asynccontextmanager

# Monitoring and metrics
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, Enum as PrometheusEnum,
    generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST, 
    start_http_server, push_to_gateway
)
import structlog
from structlog.stdlib import LoggerFactory

# Tracing
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

# Health checks
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import redis
from sqlalchemy import create_engine, text

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
class MonitoringConfig:
    """Configuration for monitoring and logging."""
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # json or text
    LOG_FILE = os.getenv("LOG_FILE", "/var/log/recipe-processing/app.log")
    LOG_ROTATION_SIZE = os.getenv("LOG_ROTATION_SIZE", "100MB")
    LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "30"))
    
    # Metrics
    METRICS_PORT = int(os.getenv("METRICS_PORT", "8080"))
    METRICS_PATH = os.getenv("METRICS_PATH", "/metrics")
    PROMETHEUS_PUSHGATEWAY_URL = os.getenv("PROMETHEUS_PUSHGATEWAY_URL")
    
    # Tracing
    JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces")
    TRACE_SAMPLING_RATE = float(os.getenv("TRACE_SAMPLING_RATE", "0.1"))
    
    # Health checks
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))  # seconds
    
    # System monitoring
    SYSTEM_METRICS_INTERVAL = int(os.getenv("SYSTEM_METRICS_INTERVAL", "10"))  # seconds
    
    # Database monitoring
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/recipe_processing")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Service info
    SERVICE_NAME = os.getenv("SERVICE_NAME", "recipe-processing-api")
    SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    HOSTNAME = socket.gethostname()

config = MonitoringConfig()

# Enums
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ComponentType(Enum):
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    FILE_SYSTEM = "file_system"
    PROCESSING_ENGINE = "processing_engine"

# Setup structured logging
def configure_logging():
    """Configure structured logging with Structlog."""
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.LOG_LEVEL.upper())
    )
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if config.LOG_FORMAT == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

configure_logging()
logger = structlog.get_logger(__name__)

# Prometheus Metrics
# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP requests', 
    ['method', 'endpoint', 'status_code']
)
REQUEST_DURATION = Histogram(
    'http_request_duration_seconds', 
    'HTTP request duration', 
    ['method', 'endpoint']
)
REQUEST_SIZE = Summary(
    'http_request_size_bytes', 
    'HTTP request size in bytes', 
    ['method', 'endpoint']
)
RESPONSE_SIZE = Summary(
    'http_response_size_bytes', 
    'HTTP response size in bytes', 
    ['method', 'endpoint']
)

# Processing metrics
PROCESSING_JOBS_TOTAL = Counter(
    'processing_jobs_total', 
    'Total processing jobs', 
    ['job_type', 'status']
)
PROCESSING_DURATION = Histogram(
    'processing_job_duration_seconds', 
    'Processing job duration', 
    ['job_type', 'format_type']
)
PROCESSING_QUEUE_SIZE = Gauge(
    'processing_queue_size', 
    'Current processing queue size'
)
ACTIVE_PROCESSING_JOBS = Gauge(
    'active_processing_jobs', 
    'Number of active processing jobs'
)

# Ingredient extraction metrics
INGREDIENTS_EXTRACTED = Counter(
    'ingredients_extracted_total', 
    'Total ingredients extracted', 
    ['format_type', 'language']
)
EXTRACTION_ACCURACY = Histogram(
    'extraction_accuracy_score', 
    'Ingredient extraction accuracy scores', 
    ['format_type']
)
OCR_CONFIDENCE = Histogram(
    'ocr_confidence_score', 
    'OCR confidence scores', 
    ['ocr_engine']
)

# System metrics
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage percentage')
SYSTEM_DISK_USAGE = Gauge('system_disk_usage_percent', 'System disk usage percentage')
SYSTEM_LOAD_AVERAGE = Gauge('system_load_average', 'System load average')

# Database metrics
DATABASE_CONNECTIONS = Gauge(
    'database_connections_active', 
    'Active database connections'
)
DATABASE_QUERY_DURATION = Histogram(
    'database_query_duration_seconds', 
    'Database query duration', 
    ['query_type']
)
DATABASE_ERRORS = Counter(
    'database_errors_total', 
    'Total database errors', 
    ['error_type']
)

# Cache metrics
CACHE_OPERATIONS = Counter(
    'cache_operations_total', 
    'Total cache operations', 
    ['operation', 'cache_type', 'result']
)
CACHE_HIT_RATIO = Gauge(
    'cache_hit_ratio', 
    'Cache hit ratio', 
    ['cache_type']
)
CACHE_SIZE_BYTES = Gauge(
    'cache_size_bytes', 
    'Cache size in bytes', 
    ['cache_type']
)

# Health check metrics
HEALTH_CHECK_STATUS = PrometheusEnum(
    'health_check_status', 
    'Health check status', 
    ['component'], 
    states=['healthy', 'degraded', 'unhealthy']
)
HEALTH_CHECK_DURATION = Histogram(
    'health_check_duration_seconds', 
    'Health check duration', 
    ['component']
)

# Service info
SERVICE_INFO = Info(
    'service_info', 
    'Service information'
)
SERVICE_INFO.info({
    'version': config.SERVICE_VERSION,
    'environment': config.ENVIRONMENT,
    'hostname': config.HOSTNAME
})

# Data Classes
@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    response_time: float
    timestamp: datetime

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    load_average: float
    network_io: Dict[str, int]
    disk_io: Dict[str, int]
    timestamp: datetime

@dataclass
class ProcessingMetrics:
    """Processing performance metrics."""
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    active_jobs: int
    queue_size: int
    average_processing_time: float
    throughput: float  # jobs per second
    timestamp: datetime

# Health Check System
class HealthChecker:
    """Comprehensive health check system."""
    
    def __init__(self):
        self.components = {}
        self.last_check_time = {}
    
    def register_component(self, name: str, check_func: callable, 
                          component_type: ComponentType, interval: int = 30):
        """Register a component for health checking."""
        self.components[name] = {
            'check_func': check_func,
            'type': component_type,
            'interval': interval
        }
        logger.info(f"Registered health check for component: {name}")
    
    async def check_component(self, name: str) -> HealthCheckResult:
        """Check health of a specific component."""
        if name not in self.components:
            return HealthCheckResult(
                component=name,
                status=HealthStatus.UNHEALTHY,
                message="Component not registered",
                details={},
                response_time=0.0,
                timestamp=datetime.utcnow()
            )
        
        component = self.components[name]
        start_time = time.time()
        
        try:
            with HEALTH_CHECK_DURATION.labels(component=name).time():
                if asyncio.iscoroutinefunction(component['check_func']):
                    result = await component['check_func']()
                else:
                    result = component['check_func']()
            
            response_time = time.time() - start_time
            
            if isinstance(result, dict):
                status = HealthStatus(result.get('status', 'healthy'))
                message = result.get('message', 'OK')
                details = result.get('details', {})
            else:
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "OK" if result else "Check failed"
                details = {}
            
            # Update metrics
            HEALTH_CHECK_STATUS.labels(component=name).state(status.value)
            
            return HealthCheckResult(
                component=name,
                status=status,
                message=message,
                details=details,
                response_time=response_time,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Health check failed for {name}", error=str(e))
            
            HEALTH_CHECK_STATUS.labels(component=name).state('unhealthy')
            
            return HealthCheckResult(
                component=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}",
                details={'error': str(e)},
                response_time=response_time,
                timestamp=datetime.utcnow()
            )
    
    async def check_all_components(self) -> Dict[str, HealthCheckResult]:
        """Check health of all registered components."""
        results = {}
        
        for name in self.components:
            results[name] = await self.check_component(name)
        
        return results
    
    async def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        results = await self.check_all_components()
        
        if not results:
            return HealthStatus.UNHEALTHY
        
        statuses = [result.status for result in results.values()]
        
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED

# System Metrics Collector
class SystemMetricsCollector:
    """Collects system performance metrics."""
    
    def __init__(self):
        self.running = False
        self.task = None
    
    def start(self):
        """Start collecting system metrics."""
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._collect_metrics())
            logger.info("System metrics collection started")
    
    def stop(self):
        """Stop collecting system metrics."""
        if self.running:
            self.running = False
            if self.task:
                self.task.cancel()
            logger.info("System metrics collection stopped")
    
    async def _collect_metrics(self):
        """Collect system metrics periodically."""
        while self.running:
            try:
                metrics = self.get_system_metrics()
                
                # Update Prometheus metrics
                SYSTEM_CPU_USAGE.set(metrics.cpu_percent)
                SYSTEM_MEMORY_USAGE.set(metrics.memory_percent)
                SYSTEM_DISK_USAGE.set(metrics.disk_percent)
                SYSTEM_LOAD_AVERAGE.set(metrics.load_average)
                
                await asyncio.sleep(config.SYSTEM_METRICS_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(config.SYSTEM_METRICS_INTERVAL)
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Load average
            if hasattr(psutil, 'getloadavg'):
                load_average = psutil.getloadavg()[0]
            else:
                load_average = 0.0
            
            # Network I/O
            network_io = psutil.net_io_counters()._asdict()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()._asdict()
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                load_average=load_average,
                network_io=network_io,
                disk_io=disk_io,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                load_average=0.0,
                network_io={},
                disk_io={},
                timestamp=datetime.utcnow()
            )

# Request Monitoring Middleware
class RequestMonitoringMiddleware:
    """Middleware for monitoring HTTP requests."""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        # Get request info
        method = request.method
        path = request.url.path
        
        # Get request size
        request_size = int(request.headers.get('content-length', 0))
        REQUEST_SIZE.labels(method=method, endpoint=path).observe(request_size)
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Get response info
        status_code = response.status_code
        response_size = int(response.headers.get('content-length', 0))
        
        # Update metrics
        REQUEST_COUNT.labels(
            method=method, 
            endpoint=path, 
            status_code=status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=method, 
            endpoint=path
        ).observe(duration)
        
        RESPONSE_SIZE.labels(
            method=method, 
            endpoint=path
        ).observe(response_size)
        
        # Log request
        logger.info(
            "HTTP request",
            method=method,
            path=path,
            status_code=status_code,
            duration=duration,
            request_size=request_size,
            response_size=response_size
        )
        
        return response

# Processing Metrics Collector
class ProcessingMetricsCollector:
    """Collects processing-specific metrics."""
    
    def __init__(self):
        self.job_start_times = {}
        self.job_counts = {
            'total': 0,
            'completed': 0,
            'failed': 0,
            'active': 0
        }
    
    def job_started(self, job_id: str, job_type: str):
        """Record job start."""
        self.job_start_times[job_id] = time.time()
        self.job_counts['total'] += 1
        self.job_counts['active'] += 1
        
        PROCESSING_JOBS_TOTAL.labels(job_type=job_type, status='started').inc()
        ACTIVE_PROCESSING_JOBS.set(self.job_counts['active'])
    
    def job_completed(self, job_id: str, job_type: str, format_type: str = None):
        """Record job completion."""
        if job_id in self.job_start_times:
            duration = time.time() - self.job_start_times[job_id]
            del self.job_start_times[job_id]
            
            PROCESSING_DURATION.labels(
                job_type=job_type, 
                format_type=format_type or 'unknown'
            ).observe(duration)
        
        self.job_counts['completed'] += 1
        self.job_counts['active'] -= 1
        
        PROCESSING_JOBS_TOTAL.labels(job_type=job_type, status='completed').inc()
        ACTIVE_PROCESSING_JOBS.set(self.job_counts['active'])
    
    def job_failed(self, job_id: str, job_type: str):
        """Record job failure."""
        if job_id in self.job_start_times:
            del self.job_start_times[job_id]
        
        self.job_counts['failed'] += 1
        self.job_counts['active'] -= 1
        
        PROCESSING_JOBS_TOTAL.labels(job_type=job_type, status='failed').inc()
        ACTIVE_PROCESSING_JOBS.set(self.job_counts['active'])
    
    def ingredient_extracted(self, format_type: str, language: str, confidence: float):
        """Record ingredient extraction."""
        INGREDIENTS_EXTRACTED.labels(
            format_type=format_type, 
            language=language
        ).inc()
        
        EXTRACTION_ACCURACY.labels(format_type=format_type).observe(confidence)
    
    def ocr_result(self, ocr_engine: str, confidence: float):
        """Record OCR result."""
        OCR_CONFIDENCE.labels(ocr_engine=ocr_engine).observe(confidence)
    
    def get_metrics(self) -> ProcessingMetrics:
        """Get current processing metrics."""
        total_processed = self.job_counts['completed'] + self.job_counts['failed']
        average_time = 0.0
        throughput = 0.0
        
        if total_processed > 0:
            # Calculate average processing time (simplified)
            average_time = sum(self.job_start_times.values()) / len(self.job_start_times) if self.job_start_times else 0.0
            
            # Calculate throughput (jobs per second in last hour - simplified)
            throughput = total_processed / 3600  # Very simplified
        
        return ProcessingMetrics(
            total_jobs=self.job_counts['total'],
            completed_jobs=self.job_counts['completed'],
            failed_jobs=self.job_counts['failed'],
            active_jobs=self.job_counts['active'],
            queue_size=0,  # Would need to be updated by queue manager
            average_processing_time=average_time,
            throughput=throughput,
            timestamp=datetime.utcnow()
        )

# Health Check Functions
async def check_database_health():
    """Check database connectivity and performance."""
    try:
        engine = create_engine(config.DATABASE_URL)
        start_time = time.time()
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        
        response_time = time.time() - start_time
        
        return {
            'status': 'healthy',
            'message': 'Database connection successful',
            'details': {
                'response_time': response_time,
                'url': config.DATABASE_URL.split('@')[1] if '@' in config.DATABASE_URL else 'hidden'
            }
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'message': f'Database connection failed: {str(e)}',
            'details': {'error': str(e)}
        }

async def check_redis_health():
    """Check Redis connectivity and performance."""
    try:
        client = redis.Redis.from_url(config.REDIS_URL)
        start_time = time.time()
        
        client.ping()
        info = client.info()
        
        response_time = time.time() - start_time
        
        return {
            'status': 'healthy',
            'message': 'Redis connection successful',
            'details': {
                'response_time': response_time,
                'used_memory': info.get('used_memory_human', 'unknown'),
                'connected_clients': info.get('connected_clients', 0)
            }
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'message': f'Redis connection failed: {str(e)}',
            'details': {'error': str(e)}
        }

def check_file_system_health():
    """Check file system health."""
    try:
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        free_percent = (disk_usage.free / disk_usage.total) * 100
        
        # Check temp directory
        temp_dir = Path('/tmp')
        temp_writable = temp_dir.exists() and os.access(temp_dir, os.W_OK)
        
        status = 'healthy'
        if free_percent < 10:
            status = 'degraded'
        if free_percent < 5 or not temp_writable:
            status = 'unhealthy'
        
        return {
            'status': status,
            'message': f'File system {status}',
            'details': {
                'free_space_percent': free_percent,
                'temp_directory_writable': temp_writable,
                'total_space': disk_usage.total,
                'free_space': disk_usage.free
            }
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'message': f'File system check failed: {str(e)}',
            'details': {'error': str(e)}
        }

# Monitoring decorators
def monitor_processing_job(job_type: str):
    """Decorator to monitor processing jobs."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            job_id = str(uuid.uuid4())
            
            try:
                processing_metrics.job_started(job_id, job_type)
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Extract format type from result if available
                format_type = getattr(result, 'format_type', 'unknown')
                processing_metrics.job_completed(job_id, job_type, format_type)
                
                return result
                
            except Exception as e:
                processing_metrics.job_failed(job_id, job_type)
                raise
                
        return wrapper
    return decorator

def track_execution_time(operation_name: str):
    """Decorator to track execution time."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.info(
                    f"Operation completed",
                    operation=operation_name,
                    duration=duration,
                    success=True
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Operation failed",
                    operation=operation_name,
                    duration=duration,
                    error=str(e),
                    success=False
                )
                raise
                
        return wrapper
    return decorator

# Global instances
health_checker = HealthChecker()
system_metrics_collector = SystemMetricsCollector()
processing_metrics = ProcessingMetricsCollector()

# Register health checks
health_checker.register_component(
    "database", 
    check_database_health, 
    ComponentType.DATABASE
)
health_checker.register_component(
    "redis", 
    check_redis_health, 
    ComponentType.CACHE
)
health_checker.register_component(
    "file_system", 
    check_file_system_health, 
    ComponentType.FILE_SYSTEM
)

# Initialize OpenTelemetry
def initialize_tracing():
    """Initialize OpenTelemetry tracing."""
    try:
        resource = Resource.create({
            "service.name": config.SERVICE_NAME,
            "service.version": config.SERVICE_VERSION,
            "deployment.environment": config.ENVIRONMENT
        })
        
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Jaeger exporter
        jaeger_exporter = JaegerExporter(
            endpoint=config.JAEGER_ENDPOINT,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        logger.info("OpenTelemetry tracing initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize tracing: {e}")

# API endpoints for monitoring
async def get_health_status():
    """Get overall health status."""
    overall_status = await health_checker.get_overall_health()
    component_results = await health_checker.check_all_components()
    
    return {
        "status": overall_status.value,
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            name: {
                "status": result.status.value,
                "message": result.message,
                "response_time": result.response_time,
                "details": result.details
            }
            for name, result in component_results.items()
        }
    }

async def get_metrics():
    """Get Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

async def get_system_status():
    """Get system performance status."""
    system_metrics = system_metrics_collector.get_system_metrics()
    processing_metrics_data = processing_metrics.get_metrics()
    
    return {
        "system": asdict(system_metrics),
        "processing": asdict(processing_metrics_data),
        "timestamp": datetime.utcnow().isoformat()
    }

# Startup and shutdown
async def start_monitoring():
    """Start monitoring services."""
    try:
        # Initialize tracing
        initialize_tracing()
        
        # Start system metrics collection
        system_metrics_collector.start()
        
        # Start Prometheus metrics server
        if config.METRICS_PORT:
            start_http_server(config.METRICS_PORT)
            logger.info(f"Prometheus metrics server started on port {config.METRICS_PORT}")
        
        logger.info("Monitoring system started")
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise

async def stop_monitoring():
    """Stop monitoring services."""
    try:
        # Stop system metrics collection
        system_metrics_collector.stop()
        
        logger.info("Monitoring system stopped")
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")

if __name__ == "__main__":
    # Test monitoring system
    async def test_monitoring():
        await start_monitoring()
        
        # Test health checks
        health_status = await get_health_status()
        print(f"Health status: {health_status}")
        
        # Test metrics collection
        system_status = await get_system_status()
        print(f"System status: {system_status}")
        
        await stop_monitoring()
    
    asyncio.run(test_monitoring())