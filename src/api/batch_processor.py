#!/usr/bin/env python3
"""
Batch Processing System for Recipe Images
Handles large-scale recipe processing with queue management, parallel processing,
and comprehensive monitoring.
"""

import os
import sys
import json
import uuid
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Database and messaging
from sqlalchemy import create_engine, Column, String, DateTime, Float, Text, Integer, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import redis
from celery import Celery
from celery.result import AsyncResult

# Monitoring and logging
import structlog
from prometheus_client import Counter, Histogram, Gauge
import psutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import processing modules
from ingredient_pipeline import IngredientExtractionPipeline
from format_specific_processor import FormatSpecificProcessor
from multilingual_measurement_handler import MultilingualMeasurementHandler
from layout_analyzer import LayoutAnalyzer

# Metrics
BATCH_JOBS_TOTAL = Counter('batch_jobs_total', 'Total batch jobs created')
BATCH_JOBS_COMPLETED = Counter('batch_jobs_completed', 'Total batch jobs completed')
BATCH_JOBS_FAILED = Counter('batch_jobs_failed', 'Total batch jobs failed')
BATCH_PROCESSING_TIME = Histogram('batch_processing_duration_seconds', 'Batch processing time')
BATCH_QUEUE_SIZE = Gauge('batch_queue_size', 'Current batch queue size')
BATCH_ACTIVE_WORKERS = Gauge('batch_active_workers', 'Active batch processing workers')
BATCH_THROUGHPUT = Counter('batch_throughput_images_total', 'Total images processed in batches')

# Configuration
class BatchConfig:
    """Configuration for batch processing."""
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/recipe_processing")
    
    # Redis/Celery
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    
    # Processing limits
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1000"))
    MAX_CONCURRENT_BATCHES = int(os.getenv("MAX_CONCURRENT_BATCHES", "5"))
    MAX_WORKERS_PER_BATCH = int(os.getenv("MAX_WORKERS_PER_BATCH", "10"))
    
    # Timeouts
    BATCH_TIMEOUT = int(os.getenv("BATCH_TIMEOUT", "7200"))  # 2 hours
    TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT", "300"))  # 5 minutes
    
    # Storage
    BATCH_STORAGE_PATH = Path(os.getenv("BATCH_STORAGE_PATH", "/tmp/batch_processing"))
    BATCH_RESULTS_PATH = Path(os.getenv("BATCH_RESULTS_PATH", "/tmp/batch_results"))
    
    # Retry configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "60"))  # seconds
    
    # Resource limits
    MAX_MEMORY_USAGE = float(os.getenv("MAX_MEMORY_USAGE", "80.0"))  # percentage
    MAX_CPU_USAGE = float(os.getenv("MAX_CPU_USAGE", "90.0"))  # percentage

config = BatchConfig()

# Ensure directories exist
config.BATCH_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
config.BATCH_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Setup logging
logger = structlog.get_logger(__name__)

# Database Models
Base = declarative_base()

class BatchJob(Base):
    __tablename__ = "batch_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="pending")
    priority = Column(String(20), nullable=False, default="normal")
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    processing_time = Column(Float)
    
    total_images = Column(Integer, default=0)
    processed_images = Column(Integer, default=0)
    failed_images = Column(Integer, default=0)
    
    progress = Column(Float, default=0.0)
    estimated_completion = Column(DateTime)
    
    configuration = Column(JSONB)
    metadata = Column(JSONB)
    error_message = Column(Text)
    
    # Relationships
    tasks = relationship("BatchTask", back_populates="batch_job")

class BatchTask(Base):
    __tablename__ = "batch_tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_job_id = Column(UUID(as_uuid=True), ForeignKey("batch_jobs.id"), nullable=False)
    
    image_path = Column(String(500), nullable=False)
    image_hash = Column(String(64), nullable=False)
    status = Column(String(50), nullable=False, default="pending")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    processing_time = Column(Float)
    
    retry_count = Column(Integer, default=0)
    worker_id = Column(String(100))
    
    results = Column(JSONB)
    error_message = Column(Text)
    
    # Relationships
    batch_job = relationship("BatchJob", back_populates="tasks")

class BatchProgress(Base):
    __tablename__ = "batch_progress"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_job_id = Column(UUID(as_uuid=True), ForeignKey("batch_jobs.id"), nullable=False)
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    processed_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    throughput = Column(Float, default=0.0)  # images per second
    
    system_metrics = Column(JSONB)  # CPU, memory, etc.

# Enums
class BatchStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class BatchPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

# Data Classes
@dataclass
class BatchJobConfig:
    """Configuration for batch job."""
    max_workers: int = 5
    chunk_size: int = 10
    timeout: int = 300
    retry_count: int = 3
    priority: BatchPriority = BatchPriority.NORMAL
    format_hint: Optional[str] = None
    language_hint: Optional[str] = None
    measurement_system_hint: Optional[str] = None
    quality_threshold: float = 0.5
    confidence_threshold: float = 0.25
    enable_caching: bool = True
    output_format: str = "json"
    include_metadata: bool = True
    processing_options: Dict[str, Any] = None

@dataclass
class BatchJobRequest:
    """Request for creating a batch job."""
    name: str
    image_paths: List[str]
    config: BatchJobConfig
    metadata: Dict[str, Any] = None

@dataclass
class BatchJobStatus:
    """Status of a batch job."""
    id: str
    name: str
    status: BatchStatus
    progress: float
    total_images: int
    processed_images: int
    failed_images: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    processing_time: Optional[float] = None
    throughput: Optional[float] = None
    error_message: Optional[str] = None

# Celery app
celery_app = Celery(
    'batch_processor',
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=config.TASK_TIMEOUT,
    task_soft_time_limit=config.TASK_TIMEOUT - 30,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=100,
)

# Database setup
engine = create_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Redis client
redis_client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)

class BatchProcessor:
    """Main batch processing system."""
    
    def __init__(self):
        self.processing_components = None
        self.active_batches = {}
        self.resource_monitor = ResourceMonitor()
        
    async def initialize(self):
        """Initialize processing components."""
        try:
            self.processing_components = {
                'pipeline': IngredientExtractionPipeline(),
                'format_processor': FormatSpecificProcessor(),
                'multilingual_handler': MultilingualMeasurementHandler(),
                'layout_analyzer': LayoutAnalyzer()
            }
            logger.info("Batch processing components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize batch processing components: {e}")
            raise
    
    async def create_batch_job(self, request: BatchJobRequest) -> str:
        """Create a new batch job."""
        BATCH_JOBS_TOTAL.inc()
        
        # Validate request
        if len(request.image_paths) > config.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum: {config.MAX_BATCH_SIZE}")
        
        # Check system resources
        if not self.resource_monitor.can_start_new_batch():
            raise ValueError("System resources insufficient for new batch")
        
        # Create batch job
        batch_id = str(uuid.uuid4())
        
        with SessionLocal() as db:
            batch_job = BatchJob(
                id=batch_id,
                name=request.name,
                status=BatchStatus.PENDING.value,
                priority=request.config.priority.value,
                total_images=len(request.image_paths),
                configuration=asdict(request.config),
                metadata=request.metadata or {}
            )
            
            db.add(batch_job)
            db.commit()
            
            # Create tasks
            tasks = []
            for image_path in request.image_paths:
                task = BatchTask(
                    batch_job_id=batch_id,
                    image_path=image_path,
                    image_hash=self._calculate_image_hash(image_path)
                )
                tasks.append(task)
            
            db.add_all(tasks)
            db.commit()
        
        # Schedule batch processing
        await self._schedule_batch_processing(batch_id)
        
        logger.info(f"Created batch job {batch_id} with {len(request.image_paths)} images")
        return batch_id
    
    async def get_batch_status(self, batch_id: str) -> BatchJobStatus:
        """Get status of a batch job."""
        with SessionLocal() as db:
            batch_job = db.query(BatchJob).filter(BatchJob.id == batch_id).first()
            if not batch_job:
                raise ValueError(f"Batch job {batch_id} not found")
            
            # Calculate throughput
            throughput = None
            if batch_job.processing_time and batch_job.processed_images:
                throughput = batch_job.processed_images / batch_job.processing_time
            
            return BatchJobStatus(
                id=str(batch_job.id),
                name=batch_job.name,
                status=BatchStatus(batch_job.status),
                progress=batch_job.progress,
                total_images=batch_job.total_images,
                processed_images=batch_job.processed_images,
                failed_images=batch_job.failed_images,
                created_at=batch_job.created_at,
                started_at=batch_job.started_at,
                completed_at=batch_job.completed_at,
                estimated_completion=batch_job.estimated_completion,
                processing_time=batch_job.processing_time,
                throughput=throughput,
                error_message=batch_job.error_message
            )
    
    async def cancel_batch_job(self, batch_id: str) -> bool:
        """Cancel a batch job."""
        with SessionLocal() as db:
            batch_job = db.query(BatchJob).filter(BatchJob.id == batch_id).first()
            if not batch_job:
                return False
            
            # Update status
            batch_job.status = BatchStatus.CANCELLED.value
            batch_job.completed_at = datetime.utcnow()
            
            # Cancel pending tasks
            pending_tasks = db.query(BatchTask).filter(
                BatchTask.batch_job_id == batch_id,
                BatchTask.status == TaskStatus.PENDING.value
            ).all()
            
            for task in pending_tasks:
                task.status = TaskStatus.FAILED.value
                task.error_message = "Batch job cancelled"
            
            db.commit()
        
        # Revoke Celery tasks
        if batch_id in self.active_batches:
            for task_id in self.active_batches[batch_id]['celery_tasks']:
                celery_app.control.revoke(task_id, terminate=True)
            del self.active_batches[batch_id]
        
        logger.info(f"Cancelled batch job {batch_id}")
        return True
    
    async def pause_batch_job(self, batch_id: str) -> bool:
        """Pause a batch job."""
        with SessionLocal() as db:
            batch_job = db.query(BatchJob).filter(BatchJob.id == batch_id).first()
            if not batch_job:
                return False
            
            batch_job.status = BatchStatus.PAUSED.value
            db.commit()
        
        logger.info(f"Paused batch job {batch_id}")
        return True
    
    async def resume_batch_job(self, batch_id: str) -> bool:
        """Resume a paused batch job."""
        with SessionLocal() as db:
            batch_job = db.query(BatchJob).filter(BatchJob.id == batch_id).first()
            if not batch_job:
                return False
            
            if batch_job.status != BatchStatus.PAUSED.value:
                return False
            
            batch_job.status = BatchStatus.RUNNING.value
            db.commit()
        
        # Reschedule processing
        await self._schedule_batch_processing(batch_id)
        
        logger.info(f"Resumed batch job {batch_id}")
        return True
    
    async def _schedule_batch_processing(self, batch_id: str):
        """Schedule batch processing using Celery."""
        # Get batch configuration
        with SessionLocal() as db:
            batch_job = db.query(BatchJob).filter(BatchJob.id == batch_id).first()
            if not batch_job:
                return
            
            config_dict = batch_job.configuration
            
            # Get pending tasks
            pending_tasks = db.query(BatchTask).filter(
                BatchTask.batch_job_id == batch_id,
                BatchTask.status == TaskStatus.PENDING.value
            ).all()
            
            # Update batch status
            batch_job.status = BatchStatus.RUNNING.value
            batch_job.started_at = datetime.utcnow()
            
            # Estimate completion time
            avg_processing_time = 120  # seconds per image (estimate)
            estimated_duration = len(pending_tasks) * avg_processing_time / config_dict.get('max_workers', 5)
            batch_job.estimated_completion = datetime.utcnow() + timedelta(seconds=estimated_duration)
            
            db.commit()
        
        # Create Celery task groups
        celery_tasks = []
        chunk_size = config_dict.get('chunk_size', 10)
        
        for i in range(0, len(pending_tasks), chunk_size):
            chunk = pending_tasks[i:i + chunk_size]
            task_ids = [str(task.id) for task in chunk]
            
            # Schedule processing chunk
            celery_task = process_batch_chunk.delay(batch_id, task_ids)
            celery_tasks.append(celery_task.id)
        
        # Track active batch
        self.active_batches[batch_id] = {
            'celery_tasks': celery_tasks,
            'start_time': datetime.utcnow()
        }
        
        # Schedule progress monitoring
        asyncio.create_task(self._monitor_batch_progress(batch_id))
    
    async def _monitor_batch_progress(self, batch_id: str):
        """Monitor batch processing progress."""
        start_time = time.time()
        
        while batch_id in self.active_batches:
            try:
                with SessionLocal() as db:
                    batch_job = db.query(BatchJob).filter(BatchJob.id == batch_id).first()
                    if not batch_job:
                        break
                    
                    # Update progress
                    if batch_job.total_images > 0:
                        progress = (batch_job.processed_images / batch_job.total_images) * 100
                        batch_job.progress = progress
                    
                    # Update processing time
                    batch_job.processing_time = time.time() - start_time
                    
                    # Check if completed
                    if batch_job.processed_images + batch_job.failed_images >= batch_job.total_images:
                        batch_job.status = BatchStatus.COMPLETED.value
                        batch_job.completed_at = datetime.utcnow()
                        
                        # Clean up
                        if batch_id in self.active_batches:
                            del self.active_batches[batch_id]
                        
                        BATCH_JOBS_COMPLETED.inc()
                        logger.info(f"Batch job {batch_id} completed")
                        break
                    
                    # Record progress
                    progress_record = BatchProgress(
                        batch_job_id=batch_id,
                        processed_count=batch_job.processed_images,
                        failed_count=batch_job.failed_images,
                        throughput=batch_job.processed_images / batch_job.processing_time if batch_job.processing_time > 0 else 0,
                        system_metrics=self.resource_monitor.get_metrics()
                    )
                    db.add(progress_record)
                    db.commit()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring batch {batch_id}: {e}")
                await asyncio.sleep(30)
    
    def _calculate_image_hash(self, image_path: str) -> str:
        """Calculate hash for image file."""
        import hashlib
        try:
            with open(image_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return hashlib.sha256(image_path.encode()).hexdigest()

class ResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self):
        self.cpu_threshold = config.MAX_CPU_USAGE
        self.memory_threshold = config.MAX_MEMORY_USAGE
    
    def can_start_new_batch(self) -> bool:
        """Check if system can handle new batch."""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            return cpu_usage < self.cpu_threshold and memory_usage < self.memory_threshold
        except Exception:
            return True  # Allow if we can't check
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}

# Celery tasks
@celery_app.task(bind=True, max_retries=config.MAX_RETRIES)
def process_batch_chunk(self, batch_id: str, task_ids: List[str]):
    """Process a chunk of batch tasks."""
    BATCH_ACTIVE_WORKERS.inc()
    
    try:
        with SessionLocal() as db:
            # Get tasks
            tasks = db.query(BatchTask).filter(BatchTask.id.in_(task_ids)).all()
            
            for task in tasks:
                try:
                    # Update task status
                    task.status = TaskStatus.RUNNING.value
                    task.started_at = datetime.utcnow()
                    task.worker_id = self.request.id
                    db.commit()
                    
                    # Process image
                    result = process_single_image(task.image_path, batch_id)
                    
                    # Update task with results
                    task.status = TaskStatus.COMPLETED.value
                    task.completed_at = datetime.utcnow()
                    task.processing_time = (task.completed_at - task.started_at).total_seconds()
                    task.results = result
                    
                    # Update batch counters
                    batch_job = db.query(BatchJob).filter(BatchJob.id == batch_id).first()
                    if batch_job:
                        batch_job.processed_images += 1
                    
                    db.commit()
                    
                    BATCH_THROUGHPUT.inc()
                    
                except Exception as e:
                    logger.error(f"Error processing task {task.id}: {e}")
                    
                    # Update task with error
                    task.status = TaskStatus.FAILED.value
                    task.completed_at = datetime.utcnow()
                    task.error_message = str(e)
                    task.retry_count += 1
                    
                    # Update batch counters
                    batch_job = db.query(BatchJob).filter(BatchJob.id == batch_id).first()
                    if batch_job:
                        batch_job.failed_images += 1
                    
                    db.commit()
                    
                    # Retry if possible
                    if task.retry_count < config.MAX_RETRIES:
                        logger.info(f"Retrying task {task.id} (attempt {task.retry_count + 1})")
                        raise self.retry(countdown=config.RETRY_DELAY, exc=e)
    
    except Exception as e:
        logger.error(f"Error processing batch chunk: {e}")
        raise
    
    finally:
        BATCH_ACTIVE_WORKERS.dec()

def process_single_image(image_path: str, batch_id: str) -> Dict[str, Any]:
    """Process a single image."""
    try:
        # Initialize processing components (cached)
        if not hasattr(process_single_image, '_components'):
            process_single_image._components = {
                'pipeline': IngredientExtractionPipeline(),
                'format_processor': FormatSpecificProcessor(),
                'multilingual_handler': MultilingualMeasurementHandler(),
                'layout_analyzer': LayoutAnalyzer()
            }
        
        components = process_single_image._components
        
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Format analysis
        format_analysis = components['format_processor'].analyze_recipe_format(image_path)
        
        # Layout analysis
        layout_analysis = components['layout_analyzer'].analyze_layout(image_path)
        
        # Process image
        results = components['pipeline'].process_image(
            image_path,
            output_dir=str(config.BATCH_RESULTS_PATH / batch_id)
        )
        
        # Extract ingredients
        ingredients = []
        if results and 'ingredients' in results:
            for ingredient_data in results['ingredients']:
                ingredients.append({
                    'ingredient_name': ingredient_data.get('ingredient_name', ''),
                    'quantity': ingredient_data.get('quantity', ''),
                    'unit': ingredient_data.get('unit', ''),
                    'confidence': ingredient_data.get('confidence', 0.0),
                    'bbox': ingredient_data.get('bbox')
                })
        
        return {
            'image_path': image_path,
            'format_analysis': {
                'detected_format': format_analysis.detected_format.value,
                'confidence': format_analysis.confidence,
                'quality_score': format_analysis.quality_score
            },
            'layout_analysis': {
                'layout_type': layout_analysis.layout_type.value,
                'confidence': layout_analysis.confidence
            },
            'ingredients': ingredients,
            'processing_metadata': {
                'processed_at': datetime.utcnow().isoformat(),
                'batch_id': batch_id
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        raise

# API functions for external use
async def create_batch_job(request: BatchJobRequest) -> str:
    """Create a new batch job."""
    processor = BatchProcessor()
    await processor.initialize()
    return await processor.create_batch_job(request)

async def get_batch_status(batch_id: str) -> BatchJobStatus:
    """Get batch job status."""
    processor = BatchProcessor()
    return await processor.get_batch_status(batch_id)

async def cancel_batch_job(batch_id: str) -> bool:
    """Cancel a batch job."""
    processor = BatchProcessor()
    return await processor.cancel_batch_job(batch_id)

async def pause_batch_job(batch_id: str) -> bool:
    """Pause a batch job."""
    processor = BatchProcessor()
    return await processor.pause_batch_job(batch_id)

async def resume_batch_job(batch_id: str) -> bool:
    """Resume a batch job."""
    processor = BatchProcessor()
    return await processor.resume_batch_job(batch_id)

if __name__ == "__main__":
    # Start Celery worker
    celery_app.start()