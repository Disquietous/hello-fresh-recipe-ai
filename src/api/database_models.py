#!/usr/bin/env python3
"""
Database Models and Integration for Recipe Processing System
Comprehensive database schema with SQLAlchemy models for storing
extracted ingredient data, processing results, and system metadata.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json

from sqlalchemy import (
    create_engine, Column, String, DateTime, Float, Text, Integer, Boolean, 
    ForeignKey, Index, CheckConstraint, UniqueConstraint, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship, backref
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_property
import structlog

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/recipe_processing")

# Setup logging
logger = structlog.get_logger(__name__)

# Base class for all models
Base = declarative_base()

# Enums
class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class FormatType(Enum):
    PRINTED_COOKBOOK = "printed_cookbook"
    HANDWRITTEN_CARD = "handwritten_card"
    DIGITAL_SCREENSHOT = "digital_screenshot"
    RECIPE_BLOG = "recipe_blog"
    MIXED_CONTENT = "mixed_content"
    UNKNOWN = "unknown"

class LayoutType(Enum):
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    BULLET_POINTS = "bullet_points"
    NUMBERED_LIST = "numbered_list"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    MIXED = "mixed"

class MeasurementSystem(Enum):
    METRIC = "metric"
    IMPERIAL = "imperial"
    MIXED = "mixed"
    TRADITIONAL_ASIAN = "traditional_asian"
    TRADITIONAL_EUROPEAN = "traditional_european"

class QualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

# Main Models
class RecipeProcessingJob(Base):
    """Main job record for recipe processing."""
    __tablename__ = "recipe_processing_jobs"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    file_path = Column(String(500))
    file_hash = Column(String(64), nullable=False, index=True)
    file_size = Column(Integer)
    
    # Status and timing
    status = Column(String(50), nullable=False, default=ProcessingStatus.PENDING.value)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    processing_time = Column(Float)
    
    # Processing results
    format_type = Column(String(50))
    layout_type = Column(String(50))
    language = Column(String(10))
    measurement_system = Column(String(20))
    quality_score = Column(Float)
    confidence_score = Column(Float)
    
    # Counts
    total_ingredients = Column(Integer, default=0)
    extracted_ingredients = Column(Integer, default=0)
    
    # Error handling
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)
    error_details = Column(JSONB)
    
    # Metadata
    processing_metadata = Column(JSONB)
    user_metadata = Column(JSONB)
    
    # User info
    user_id = Column(String(100))
    api_key_id = Column(String(100))
    
    # Relationships
    ingredients = relationship("ExtractedIngredient", back_populates="job", cascade="all, delete-orphan")
    analysis_results = relationship("AnalysisResult", back_populates="job", cascade="all, delete-orphan")
    quality_metrics = relationship("QualityMetric", back_populates="job", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_job_status_created', 'status', 'created_at'),
        Index('idx_job_user_created', 'user_id', 'created_at'),
        Index('idx_job_file_hash', 'file_hash'),
        Index('idx_job_format_type', 'format_type'),
        CheckConstraint('quality_score >= 0 AND quality_score <= 1', name='check_quality_score_range'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_confidence_score_range'),
    )
    
    @hybrid_property
    def success_rate(self):
        """Calculate success rate of ingredient extraction."""
        if self.total_ingredients == 0:
            return 0.0
        return self.extracted_ingredients / self.total_ingredients
    
    @hybrid_property
    def is_completed(self):
        """Check if job is completed."""
        return self.status in [ProcessingStatus.COMPLETED.value, ProcessingStatus.FAILED.value]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': str(self.id),
            'filename': self.filename,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'processing_time': self.processing_time,
            'format_type': self.format_type,
            'language': self.language,
            'quality_score': self.quality_score,
            'confidence_score': self.confidence_score,
            'total_ingredients': self.total_ingredients,
            'extracted_ingredients': self.extracted_ingredients
        }

class ExtractedIngredient(Base):
    """Extracted ingredient data."""
    __tablename__ = "extracted_ingredients"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("recipe_processing_jobs.id"), nullable=False)
    
    # Ingredient data
    ingredient_name = Column(String(200), nullable=False)
    ingredient_name_normalized = Column(String(200))
    ingredient_name_en = Column(String(200))
    ingredient_category = Column(String(100))
    
    # Quantity and units
    quantity = Column(String(50))
    quantity_normalized = Column(Float)
    unit = Column(String(50))
    unit_normalized = Column(String(50))
    
    # Preparation and notes
    preparation = Column(String(200))
    preparation_normalized = Column(String(200))
    notes = Column(Text)
    
    # Detection metadata
    confidence = Column(Float, nullable=False)
    detection_method = Column(String(50))
    source_region = Column(String(100))  # e.g., "text_line", "bullet_point"
    
    # Bounding box coordinates
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    
    # Language and measurement system
    language = Column(String(10))
    measurement_system = Column(String(20))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Validation flags
    is_validated = Column(Boolean, default=False)
    is_corrected = Column(Boolean, default=False)
    validation_score = Column(Float)
    
    # Relationships
    job = relationship("RecipeProcessingJob", back_populates="ingredients")
    
    # Indexes
    __table_args__ = (
        Index('idx_ingredient_job_id', 'job_id'),
        Index('idx_ingredient_name', 'ingredient_name'),
        Index('idx_ingredient_category', 'ingredient_category'),
        Index('idx_ingredient_confidence', 'confidence'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_ingredient_confidence_range'),
    )
    
    @hybrid_property
    def bbox(self) -> Optional[Dict[str, int]]:
        """Get bounding box as dictionary."""
        if all(coord is not None for coord in [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2]):
            return {
                'x1': self.bbox_x1,
                'y1': self.bbox_y1,
                'x2': self.bbox_x2,
                'y2': self.bbox_y2
            }
        return None
    
    @bbox.setter
    def bbox(self, value: Optional[Dict[str, int]]):
        """Set bounding box from dictionary."""
        if value:
            self.bbox_x1 = value.get('x1')
            self.bbox_y1 = value.get('y1')
            self.bbox_x2 = value.get('x2')
            self.bbox_y2 = value.get('y2')
        else:
            self.bbox_x1 = self.bbox_y1 = self.bbox_x2 = self.bbox_y2 = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': str(self.id),
            'ingredient_name': self.ingredient_name,
            'ingredient_name_en': self.ingredient_name_en,
            'quantity': self.quantity,
            'unit': self.unit,
            'unit_normalized': self.unit_normalized,
            'preparation': self.preparation,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'language': self.language,
            'measurement_system': self.measurement_system
        }

class AnalysisResult(Base):
    """Analysis results for different processing stages."""
    __tablename__ = "analysis_results"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("recipe_processing_jobs.id"), nullable=False)
    
    # Analysis type
    analysis_type = Column(String(50), nullable=False)  # format, layout, language, etc.
    
    # Results
    primary_result = Column(String(100))
    confidence = Column(Float)
    secondary_results = Column(JSONB)
    
    # Detailed analysis data
    analysis_data = Column(JSONB)
    processing_time = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    job = relationship("RecipeProcessingJob", back_populates="analysis_results")
    
    # Indexes
    __table_args__ = (
        Index('idx_analysis_job_type', 'job_id', 'analysis_type'),
        Index('idx_analysis_type', 'analysis_type'),
        UniqueConstraint('job_id', 'analysis_type', name='uq_job_analysis_type'),
    )

class QualityMetric(Base):
    """Quality metrics for processing results."""
    __tablename__ = "quality_metrics"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("recipe_processing_jobs.id"), nullable=False)
    
    # Metric type
    metric_type = Column(String(50), nullable=False)  # detection, ocr, parsing, overall
    
    # Quality scores
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    accuracy = Column(Float)
    
    # Detailed metrics
    metric_data = Column(JSONB)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    job = relationship("RecipeProcessingJob", back_populates="quality_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_quality_job_type', 'job_id', 'metric_type'),
        Index('idx_quality_f1_score', 'f1_score'),
        UniqueConstraint('job_id', 'metric_type', name='uq_job_metric_type'),
    )

# Caching Models
class ProcessingCache(Base):
    """Cache for processing results."""
    __tablename__ = "processing_cache"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)
    
    # Cache data
    cache_key = Column(String(100), nullable=False)
    cache_data = Column(JSONB, nullable=False)
    
    # Metadata
    file_size = Column(Integer)
    format_type = Column(String(50))
    processing_options_hash = Column(String(64))
    
    # Usage tracking
    hit_count = Column(Integer, default=0)
    last_hit_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime)
    
    # Indexes
    __table_args__ = (
        Index('idx_cache_key', 'cache_key'),
        Index('idx_cache_expires', 'expires_at'),
        Index('idx_cache_hit_count', 'hit_count'),
    )
    
    @hybrid_property
    def is_expired(self):
        """Check if cache entry is expired."""
        return self.expires_at and self.expires_at < datetime.utcnow()
    
    def update_hit_count(self):
        """Update hit count and last hit time."""
        self.hit_count += 1
        self.last_hit_at = datetime.utcnow()

# Batch Processing Models
class BatchJob(Base):
    """Batch processing job."""
    __tablename__ = "batch_jobs"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Status and timing
    status = Column(String(50), nullable=False, default=ProcessingStatus.PENDING.value)
    priority = Column(String(20), nullable=False, default="normal")
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    processing_time = Column(Float)
    
    # Progress tracking
    total_jobs = Column(Integer, default=0)
    completed_jobs = Column(Integer, default=0)
    failed_jobs = Column(Integer, default=0)
    progress = Column(Float, default=0.0)
    
    # Estimates
    estimated_completion = Column(DateTime)
    estimated_processing_time = Column(Float)
    
    # Configuration
    configuration = Column(JSONB)
    
    # User info
    user_id = Column(String(100))
    
    # Error handling
    error_message = Column(Text)
    error_count = Column(Integer, default=0)
    
    # Relationships
    tasks = relationship("BatchTask", back_populates="batch_job", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_batch_status_created', 'status', 'created_at'),
        Index('idx_batch_user_created', 'user_id', 'created_at'),
        Index('idx_batch_priority', 'priority'),
    )

class BatchTask(Base):
    """Individual task within a batch job."""
    __tablename__ = "batch_tasks"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_job_id = Column(UUID(as_uuid=True), ForeignKey("batch_jobs.id"), nullable=False)
    
    # Task data
    task_order = Column(Integer, nullable=False)
    image_path = Column(String(500), nullable=False)
    image_hash = Column(String(64), nullable=False)
    
    # Status and timing
    status = Column(String(50), nullable=False, default=ProcessingStatus.PENDING.value)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    processing_time = Column(Float)
    
    # Worker info
    worker_id = Column(String(100))
    worker_hostname = Column(String(100))
    
    # Results
    results = Column(JSONB)
    
    # Error handling
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)
    error_details = Column(JSONB)
    
    # Relationships
    batch_job = relationship("BatchJob", back_populates="tasks")
    
    # Indexes
    __table_args__ = (
        Index('idx_batch_task_batch_order', 'batch_job_id', 'task_order'),
        Index('idx_batch_task_status', 'status'),
        Index('idx_batch_task_image_hash', 'image_hash'),
    )

# System Models
class SystemMetrics(Base):
    """System performance metrics."""
    __tablename__ = "system_metrics"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Metrics
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))
    
    # Context
    hostname = Column(String(100))
    process_id = Column(String(50))
    
    # Detailed data
    metric_data = Column(JSONB)
    
    # Timestamp
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_metrics_name_recorded', 'metric_name', 'recorded_at'),
        Index('idx_metrics_hostname', 'hostname'),
    )

class APIUsage(Base):
    """API usage tracking."""
    __tablename__ = "api_usage"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Request info
    endpoint = Column(String(200), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    
    # User info
    user_id = Column(String(100))
    api_key_id = Column(String(100))
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Timing
    request_time = Column(Float)
    response_time = Column(Float)
    
    # Data
    request_size = Column(Integer)
    response_size = Column(Integer)
    
    # Errors
    error_message = Column(Text)
    error_type = Column(String(100))
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_api_usage_endpoint_timestamp', 'endpoint', 'timestamp'),
        Index('idx_api_usage_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_api_usage_status_code', 'status_code'),
    )

# User and Authentication Models
class APIKey(Base):
    """API key management."""
    __tablename__ = "api_keys"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash = Column(String(64), nullable=False, unique=True, index=True)
    
    # Key info
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # User info
    user_id = Column(String(100), nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Limits
    rate_limit_per_minute = Column(Integer, default=60)
    rate_limit_per_hour = Column(Integer, default=1000)
    rate_limit_per_day = Column(Integer, default=10000)
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime)
    
    # Indexes
    __table_args__ = (
        Index('idx_api_key_user_id', 'user_id'),
        Index('idx_api_key_active', 'is_active'),
    )
    
    @hybrid_property
    def is_expired(self):
        """Check if API key is expired."""
        return self.expires_at and self.expires_at < datetime.utcnow()

# Database Management Functions
class DatabaseManager:
    """Database management utilities."""
    
    def __init__(self, database_url: str = DATABASE_URL):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def cleanup_old_cache(self, days: int = 7):
        """Clean up old cache entries."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            with self.get_session() as session:
                # Remove expired cache entries
                expired_count = session.query(ProcessingCache).filter(
                    ProcessingCache.expires_at < cutoff_date
                ).delete()
                
                # Remove old, unused cache entries
                unused_count = session.query(ProcessingCache).filter(
                    ProcessingCache.last_hit_at < cutoff_date,
                    ProcessingCache.hit_count < 5
                ).delete()
                
                session.commit()
                
                logger.info(f"Cleaned up {expired_count} expired and {unused_count} unused cache entries")
                
        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")
            raise
    
    def cleanup_old_jobs(self, days: int = 30):
        """Clean up old completed jobs."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            with self.get_session() as session:
                # Remove old completed jobs
                old_jobs = session.query(RecipeProcessingJob).filter(
                    RecipeProcessingJob.completed_at < cutoff_date,
                    RecipeProcessingJob.status.in_([ProcessingStatus.COMPLETED.value, ProcessingStatus.FAILED.value])
                ).all()
                
                for job in old_jobs:
                    session.delete(job)
                
                session.commit()
                
                logger.info(f"Cleaned up {len(old_jobs)} old jobs")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {e}")
            raise

# Event listeners for automatic updates
@event.listens_for(ProcessingCache, 'before_update')
def update_cache_timestamp(mapper, connection, target):
    """Update cache timestamp on hit."""
    target.updated_at = datetime.utcnow()

@event.listens_for(ExtractedIngredient, 'before_update')
def update_ingredient_timestamp(mapper, connection, target):
    """Update ingredient timestamp on modification."""
    target.updated_at = datetime.utcnow()

# Database initialization
def init_database(database_url: str = DATABASE_URL):
    """Initialize database with tables."""
    db_manager = DatabaseManager(database_url)
    db_manager.create_tables()
    return db_manager

# Utility functions
def get_db_session():
    """Get database session (dependency injection)."""
    db_manager = DatabaseManager()
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    # Initialize database
    print("Initializing database...")
    db_manager = init_database()
    print("Database initialized successfully!")
    
    # Create a test session
    with db_manager.get_session() as session:
        # Test basic operations
        job = RecipeProcessingJob(
            filename="test_recipe.jpg",
            file_hash="test_hash_123",
            status=ProcessingStatus.PENDING.value
        )
        session.add(job)
        session.commit()
        
        print(f"Created test job: {job.id}")
        
        # Clean up
        session.delete(job)
        session.commit()
        
        print("Test completed successfully!")