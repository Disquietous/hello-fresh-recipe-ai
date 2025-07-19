#!/usr/bin/env python3
"""
Production Recipe Processing API
FastAPI application for recipe image processing and ingredient extraction.
Supports single image processing, batch processing, and database integration.
"""

import os
import sys
import json
import uuid
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager

# FastAPI and related imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

# Database and caching
from sqlalchemy import create_engine, Column, String, DateTime, Float, Text, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
import redis
from redis import Redis

# Monitoring and logging
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our processing modules
from ingredient_pipeline import IngredientExtractionPipeline
from format_specific_processor import FormatSpecificProcessor
from multilingual_measurement_handler import MultilingualMeasurementHandler
from layout_analyzer import LayoutAnalyzer
from comprehensive_evaluation_system import ComprehensiveEvaluationSystem

# Metrics
REQUEST_COUNT = Counter('recipe_processing_requests_total', 'Total recipe processing requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('recipe_processing_request_duration_seconds', 'Recipe processing request duration')
PROCESSING_TIME = Histogram('recipe_processing_duration_seconds', 'Recipe processing time', ['format_type'])
CACHE_HITS = Counter('recipe_processing_cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('recipe_processing_cache_misses_total', 'Total cache misses')
ACTIVE_JOBS = Gauge('recipe_processing_active_jobs', 'Number of active processing jobs')
ERROR_COUNT = Counter('recipe_processing_errors_total', 'Total processing errors', ['error_type'])

# Database Models
Base = declarative_base()

class RecipeProcessingJob(Base):
    __tablename__ = "recipe_processing_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    processing_time = Column(Float)
    format_type = Column(String(50))
    language = Column(String(10))
    measurement_system = Column(String(20))
    quality_score = Column(Float)
    confidence_score = Column(Float)
    error_message = Column(Text)
    results = Column(JSONB)
    metadata = Column(JSONB)

class ExtractedIngredient(Base):
    __tablename__ = "extracted_ingredients"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    ingredient_name = Column(String(200), nullable=False)
    ingredient_name_en = Column(String(200))
    quantity = Column(String(50))
    unit = Column(String(50))
    unit_normalized = Column(String(50))
    preparation = Column(String(200))
    confidence = Column(Float)
    bbox = Column(JSONB)  # Bounding box coordinates
    created_at = Column(DateTime, default=datetime.utcnow)

class ProcessingCache(Base):
    __tablename__ = "processing_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)
    results = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    accessed_at = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=1)

# Pydantic Models
class RecipeProcessingRequest(BaseModel):
    """Request model for recipe processing."""
    format_hint: Optional[str] = Field(None, description="Hint for recipe format (e.g., 'cookbook', 'handwritten', 'digital')")
    language_hint: Optional[str] = Field(None, description="Hint for recipe language (e.g., 'en', 'es', 'fr')")
    measurement_system_hint: Optional[str] = Field(None, description="Hint for measurement system (e.g., 'metric', 'imperial')")
    quality_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum quality threshold")
    confidence_threshold: Optional[float] = Field(0.25, ge=0.0, le=1.0, description="Minimum confidence threshold")
    enable_caching: bool = Field(True, description="Enable result caching")
    processing_options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional processing options")

class BatchProcessingRequest(BaseModel):
    """Request model for batch processing."""
    job_name: str = Field(..., description="Name for the batch job")
    processing_options: RecipeProcessingRequest = Field(default_factory=RecipeProcessingRequest)
    max_concurrent: int = Field(5, ge=1, le=20, description="Maximum concurrent processing jobs")
    priority: str = Field("normal", description="Processing priority (low, normal, high)")

class IngredientResponse(BaseModel):
    """Response model for extracted ingredient."""
    ingredient_name: str
    ingredient_name_en: Optional[str]
    quantity: Optional[str]
    unit: Optional[str]
    unit_normalized: Optional[str]
    preparation: Optional[str]
    confidence: float
    bbox: Optional[Dict[str, int]]

class RecipeProcessingResponse(BaseModel):
    """Response model for recipe processing."""
    job_id: str
    status: str
    filename: str
    processing_time: Optional[float]
    format_type: Optional[str]
    language: Optional[str]
    measurement_system: Optional[str]
    quality_score: Optional[float]
    confidence_score: Optional[float]
    ingredients: List[IngredientResponse]
    metadata: Dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime]

class BatchProcessingResponse(BaseModel):
    """Response model for batch processing."""
    batch_id: str
    job_name: str
    status: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    progress: float
    created_at: datetime
    estimated_completion: Optional[datetime]

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    timestamp: datetime
    database_status: str
    cache_status: str
    processing_queue_size: int
    system_info: Dict[str, Any]

# Configuration
class Config:
    """Configuration settings."""
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/recipe_processing")
    
    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # API
    API_VERSION = "1.0.0"
    API_TITLE = "Recipe Processing API"
    API_DESCRIPTION = "Production API for recipe image processing and ingredient extraction"
    
    # Authentication
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    
    # Processing
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    PROCESSING_TIMEOUT = 300  # 5 minutes
    
    # Caching
    CACHE_TTL = 3600  # 1 hour
    
    # Monitoring
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    
    # Batch processing
    MAX_BATCH_SIZE = 100
    BATCH_PROCESSING_TIMEOUT = 3600  # 1 hour

config = Config()

# Initialize Sentry for error tracking
if config.SENTRY_DSN:
    sentry_sdk.init(
        dsn=config.SENTRY_DSN,
        integrations=[
            FastApiIntegration(auto_enabling_integrations=False),
            SqlalchemyIntegration(),
        ],
        traces_sample_rate=0.1,
        environment=os.getenv("ENVIRONMENT", "development")
    )

# Setup structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Database setup
engine = create_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Redis setup
redis_client = Redis.from_url(config.REDIS_URL, decode_responses=True)

# Global instances
processing_pipeline = None
format_processor = None
multilingual_handler = None
layout_analyzer = None
evaluation_system = None

# Background job queue
processing_queue = asyncio.Queue()
batch_jobs = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    global processing_pipeline, format_processor, multilingual_handler, layout_analyzer, evaluation_system
    
    # Initialize processing components
    logger.info("Initializing processing components...")
    
    try:
        processing_pipeline = IngredientExtractionPipeline()
        format_processor = FormatSpecificProcessor()
        multilingual_handler = MultilingualMeasurementHandler()
        layout_analyzer = LayoutAnalyzer()
        evaluation_system = ComprehensiveEvaluationSystem()
        
        logger.info("Processing components initialized successfully")
        
        # Start background workers
        asyncio.create_task(process_queue_worker())
        asyncio.create_task(batch_processing_worker())
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize processing components: {e}")
        raise
    finally:
        logger.info("Shutting down processing components...")

# FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication
security = HTTPBearer()

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token."""
    # Implement your token verification logic here
    # For now, we'll just check if token exists
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    return hashlib.sha256(file_content).hexdigest()

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file."""
    if file.size > config.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed types: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )

async def get_cached_result(file_hash: str, db: Session) -> Optional[Dict[str, Any]]:
    """Get cached processing result."""
    try:
        # Check Redis cache first
        cached_result = redis_client.get(f"recipe_result:{file_hash}")
        if cached_result:
            CACHE_HITS.inc()
            return json.loads(cached_result)
        
        # Check database cache
        cache_entry = db.query(ProcessingCache).filter(
            ProcessingCache.file_hash == file_hash
        ).first()
        
        if cache_entry:
            # Update access info
            cache_entry.accessed_at = datetime.utcnow()
            cache_entry.access_count += 1
            db.commit()
            
            # Store in Redis for faster access
            redis_client.setex(
                f"recipe_result:{file_hash}", 
                config.CACHE_TTL, 
                json.dumps(cache_entry.results)
            )
            
            CACHE_HITS.inc()
            return cache_entry.results
        
        CACHE_MISSES.inc()
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving cached result: {e}")
        CACHE_MISSES.inc()
        return None

async def store_cached_result(file_hash: str, results: Dict[str, Any], db: Session) -> None:
    """Store processing result in cache."""
    try:
        # Store in Redis
        redis_client.setex(
            f"recipe_result:{file_hash}", 
            config.CACHE_TTL, 
            json.dumps(results)
        )
        
        # Store in database
        cache_entry = ProcessingCache(
            file_hash=file_hash,
            results=results
        )
        db.add(cache_entry)
        db.commit()
        
    except Exception as e:
        logger.error(f"Error storing cached result: {e}")

async def process_recipe_image(
    file_content: bytes,
    filename: str,
    file_hash: str,
    request: RecipeProcessingRequest,
    db: Session
) -> RecipeProcessingResponse:
    """Process recipe image and extract ingredients."""
    
    start_time = datetime.utcnow()
    job_id = str(uuid.uuid4())
    
    # Create job record
    job = RecipeProcessingJob(
        id=job_id,
        filename=filename,
        file_hash=file_hash,
        status="processing",
        started_at=start_time,
        metadata=request.processing_options
    )
    db.add(job)
    db.commit()
    
    try:
        ACTIVE_JOBS.inc()
        
        # Save temporary file
        temp_path = Path(f"/tmp/{job_id}_{filename}")
        temp_path.write_bytes(file_content)
        
        # Format analysis
        format_analysis = format_processor.analyze_recipe_format(str(temp_path))
        
        # Layout analysis
        layout_analysis = layout_analyzer.analyze_layout(str(temp_path))
        
        # Language detection
        # Use OCR to get sample text for language detection
        sample_text = ""  # This would be populated by OCR
        
        if sample_text:
            language_result = multilingual_handler.detect_language(sample_text)
            measurement_result = multilingual_handler.detect_measurement_system(
                sample_text, language_result.primary_language
            )
        else:
            # Use hints or defaults
            language_result = None
            measurement_result = None
        
        # Apply format-specific preprocessing
        import cv2
        image = cv2.imread(str(temp_path))
        processed_image = format_processor.apply_format_specific_preprocessing(
            image, format_analysis
        )
        
        # Save processed image
        processed_path = Path(f"/tmp/{job_id}_processed.jpg")
        cv2.imwrite(str(processed_path), processed_image)
        
        # Extract ingredients using pipeline
        pipeline_results = processing_pipeline.process_image(
            str(processed_path),
            output_dir=f"/tmp/{job_id}_output"
        )
        
        # Process results
        ingredients = []
        if pipeline_results and 'ingredients' in pipeline_results:
            for ingredient_data in pipeline_results['ingredients']:
                # Parse multilingual ingredient if language detected
                if language_result and measurement_result:
                    parsed_ingredient = multilingual_handler.parse_multilingual_ingredient(
                        ingredient_data.get('raw_text', ''),
                        language_result.primary_language,
                        measurement_result.primary_system
                    )
                    
                    ingredient = IngredientResponse(
                        ingredient_name=parsed_ingredient.ingredient_name,
                        ingredient_name_en=parsed_ingredient.ingredient_name_en,
                        quantity=parsed_ingredient.quantity,
                        unit=parsed_ingredient.unit,
                        unit_normalized=parsed_ingredient.unit_normalized,
                        preparation=parsed_ingredient.preparation,
                        confidence=parsed_ingredient.confidence,
                        bbox=ingredient_data.get('bbox')
                    )
                else:
                    ingredient = IngredientResponse(
                        ingredient_name=ingredient_data.get('ingredient_name', ''),
                        ingredient_name_en=ingredient_data.get('ingredient_name', ''),
                        quantity=ingredient_data.get('quantity', ''),
                        unit=ingredient_data.get('unit', ''),
                        unit_normalized=ingredient_data.get('unit_normalized', ''),
                        preparation=ingredient_data.get('preparation', ''),
                        confidence=ingredient_data.get('confidence', 0.0),
                        bbox=ingredient_data.get('bbox')
                    )
                
                ingredients.append(ingredient)
                
                # Store in database
                db_ingredient = ExtractedIngredient(
                    job_id=job_id,
                    ingredient_name=ingredient.ingredient_name,
                    ingredient_name_en=ingredient.ingredient_name_en,
                    quantity=ingredient.quantity,
                    unit=ingredient.unit,
                    unit_normalized=ingredient.unit_normalized,
                    preparation=ingredient.preparation,
                    confidence=ingredient.confidence,
                    bbox=ingredient.bbox
                )
                db.add(db_ingredient)
        
        # Calculate metrics
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        quality_score = format_analysis.quality_score
        confidence_score = np.mean([ing.confidence for ing in ingredients]) if ingredients else 0.0
        
        # Update job record
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.processing_time = processing_time
        job.format_type = format_analysis.detected_format.value
        job.language = language_result.primary_language.value if language_result else None
        job.measurement_system = measurement_result.primary_system.value if measurement_result else None
        job.quality_score = quality_score
        job.confidence_score = confidence_score
        job.results = {
            'ingredients': [ing.dict() for ing in ingredients],
            'format_analysis': {
                'detected_format': format_analysis.detected_format.value,
                'confidence': format_analysis.confidence,
                'quality_score': format_analysis.quality_score
            },
            'layout_analysis': {
                'layout_type': layout_analysis.layout_type.value,
                'confidence': layout_analysis.confidence
            }
        }
        
        db.commit()
        
        # Update metrics
        PROCESSING_TIME.labels(format_type=format_analysis.detected_format.value).observe(processing_time)
        
        # Clean up temporary files
        temp_path.unlink(missing_ok=True)
        processed_path.unlink(missing_ok=True)
        
        response = RecipeProcessingResponse(
            job_id=job_id,
            status="completed",
            filename=filename,
            processing_time=processing_time,
            format_type=format_analysis.detected_format.value,
            language=language_result.primary_language.value if language_result else None,
            measurement_system=measurement_result.primary_system.value if measurement_result else None,
            quality_score=quality_score,
            confidence_score=confidence_score,
            ingredients=ingredients,
            metadata={
                'format_analysis': format_analysis.dict(),
                'layout_analysis': layout_analysis.dict(),
                'language_result': language_result.dict() if language_result else None,
                'measurement_result': measurement_result.dict() if measurement_result else None
            },
            created_at=start_time,
            completed_at=datetime.utcnow()
        )
        
        # Cache result if enabled
        if request.enable_caching:
            await store_cached_result(file_hash, response.dict(), db)
        
        return response
        
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        logger.error(f"Error processing recipe image: {e}")
        
        # Update job record with error
        job.status = "failed"
        job.completed_at = datetime.utcnow()
        job.error_message = str(e)
        job.processing_time = (datetime.utcnow() - start_time).total_seconds()
        db.commit()
        
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        ACTIVE_JOBS.dec()

async def process_queue_worker():
    """Background worker for processing queue."""
    while True:
        try:
            # Get job from queue
            job_data = await processing_queue.get()
            
            # Process job
            await process_recipe_image(**job_data)
            
            # Mark task as done
            processing_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error in queue worker: {e}")
            await asyncio.sleep(1)

async def batch_processing_worker():
    """Background worker for batch processing."""
    while True:
        try:
            # Check for batch jobs
            for batch_id, batch_info in batch_jobs.items():
                if batch_info['status'] == 'processing':
                    # Process batch
                    await process_batch(batch_id, batch_info)
            
            await asyncio.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in batch processing worker: {e}")
            await asyncio.sleep(10)

async def process_batch(batch_id: str, batch_info: Dict[str, Any]):
    """Process a batch of recipe images."""
    # Implementation for batch processing
    pass

# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    REQUEST_COUNT.labels(method="GET", endpoint="/health").inc()
    
    # Check database
    try:
        db.execute("SELECT 1")
        database_status = "healthy"
    except Exception:
        database_status = "unhealthy"
    
    # Check Redis
    try:
        redis_client.ping()
        cache_status = "healthy"
    except Exception:
        cache_status = "unhealthy"
    
    return HealthResponse(
        status="healthy" if database_status == "healthy" and cache_status == "healthy" else "unhealthy",
        version=config.API_VERSION,
        timestamp=datetime.utcnow(),
        database_status=database_status,
        cache_status=cache_status,
        processing_queue_size=processing_queue.qsize(),
        system_info={
            "active_jobs": ACTIVE_JOBS._value.get(),
            "total_requests": REQUEST_COUNT._value.sum(),
            "cache_hit_rate": CACHE_HITS._value.sum() / (CACHE_HITS._value.sum() + CACHE_MISSES._value.sum()) if (CACHE_HITS._value.sum() + CACHE_MISSES._value.sum()) > 0 else 0
        }
    )

@app.post("/process", response_model=RecipeProcessingResponse)
async def process_recipe(
    file: UploadFile = File(...),
    request: RecipeProcessingRequest = Depends(),
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Process a single recipe image."""
    REQUEST_COUNT.labels(method="POST", endpoint="/process").inc()
    
    with REQUEST_DURATION.time():
        # Validate file
        validate_file(file)
        
        # Read file content
        file_content = await file.read()
        file_hash = calculate_file_hash(file_content)
        
        # Check cache if enabled
        if request.enable_caching:
            cached_result = await get_cached_result(file_hash, db)
            if cached_result:
                return RecipeProcessingResponse(**cached_result)
        
        # Process image
        return await process_recipe_image(
            file_content, file.filename, file_hash, request, db
        )

@app.post("/batch", response_model=BatchProcessingResponse)
async def start_batch_processing(
    files: List[UploadFile] = File(...),
    request: BatchProcessingRequest = Depends(),
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Start batch processing of multiple recipe images."""
    REQUEST_COUNT.labels(method="POST", endpoint="/batch").inc()
    
    if len(files) > config.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Too many files. Maximum allowed: {config.MAX_BATCH_SIZE}"
        )
    
    batch_id = str(uuid.uuid4())
    
    # Create batch job
    batch_jobs[batch_id] = {
        'batch_id': batch_id,
        'job_name': request.job_name,
        'status': 'processing',
        'total_jobs': len(files),
        'completed_jobs': 0,
        'failed_jobs': 0,
        'files': files,
        'request': request,
        'created_at': datetime.utcnow(),
        'estimated_completion': datetime.utcnow() + timedelta(minutes=len(files) * 2)
    }
    
    return BatchProcessingResponse(
        batch_id=batch_id,
        job_name=request.job_name,
        status='processing',
        total_jobs=len(files),
        completed_jobs=0,
        failed_jobs=0,
        progress=0.0,
        created_at=datetime.utcnow(),
        estimated_completion=datetime.utcnow() + timedelta(minutes=len(files) * 2)
    )

@app.get("/batch/{batch_id}", response_model=BatchProcessingResponse)
async def get_batch_status(
    batch_id: str,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get batch processing status."""
    REQUEST_COUNT.labels(method="GET", endpoint="/batch/{batch_id}").inc()
    
    if batch_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    batch_info = batch_jobs[batch_id]
    progress = batch_info['completed_jobs'] / batch_info['total_jobs'] * 100
    
    return BatchProcessingResponse(
        batch_id=batch_id,
        job_name=batch_info['job_name'],
        status=batch_info['status'],
        total_jobs=batch_info['total_jobs'],
        completed_jobs=batch_info['completed_jobs'],
        failed_jobs=batch_info['failed_jobs'],
        progress=progress,
        created_at=batch_info['created_at'],
        estimated_completion=batch_info.get('estimated_completion')
    )

@app.get("/job/{job_id}", response_model=RecipeProcessingResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get job processing status."""
    REQUEST_COUNT.labels(method="GET", endpoint="/job/{job_id}").inc()
    
    job = db.query(RecipeProcessingJob).filter(RecipeProcessingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get ingredients
    ingredients = db.query(ExtractedIngredient).filter(ExtractedIngredient.job_id == job_id).all()
    
    ingredient_responses = [
        IngredientResponse(
            ingredient_name=ing.ingredient_name,
            ingredient_name_en=ing.ingredient_name_en,
            quantity=ing.quantity,
            unit=ing.unit,
            unit_normalized=ing.unit_normalized,
            preparation=ing.preparation,
            confidence=ing.confidence,
            bbox=ing.bbox
        ) for ing in ingredients
    ]
    
    return RecipeProcessingResponse(
        job_id=str(job.id),
        status=job.status,
        filename=job.filename,
        processing_time=job.processing_time,
        format_type=job.format_type,
        language=job.language,
        measurement_system=job.measurement_system,
        quality_score=job.quality_score,
        confidence_score=job.confidence_score,
        ingredients=ingredient_responses,
        metadata=job.metadata or {},
        created_at=job.created_at,
        completed_at=job.completed_at
    )

@app.get("/jobs", response_model=List[RecipeProcessingResponse])
async def list_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """List processing jobs."""
    REQUEST_COUNT.labels(method="GET", endpoint="/jobs").inc()
    
    query = db.query(RecipeProcessingJob)
    
    if status:
        query = query.filter(RecipeProcessingJob.status == status)
    
    jobs = query.order_by(RecipeProcessingJob.created_at.desc()).offset(skip).limit(limit).all()
    
    return [
        RecipeProcessingResponse(
            job_id=str(job.id),
            status=job.status,
            filename=job.filename,
            processing_time=job.processing_time,
            format_type=job.format_type,
            language=job.language,
            measurement_system=job.measurement_system,
            quality_score=job.quality_score,
            confidence_score=job.confidence_score,
            ingredients=[],  # Don't include ingredients in list view
            metadata=job.metadata or {},
            created_at=job.created_at,
            completed_at=job.completed_at
        ) for job in jobs
    ]

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.delete("/cache")
async def clear_cache(
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Clear processing cache."""
    REQUEST_COUNT.labels(method="DELETE", endpoint="/cache").inc()
    
    try:
        # Clear Redis cache
        redis_client.flushdb()
        
        # Clear database cache
        db.query(ProcessingCache).delete()
        db.commit()
        
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

if __name__ == "__main__":
    uvicorn.run(
        "recipe_processing_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )