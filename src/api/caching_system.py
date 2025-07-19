#!/usr/bin/env python3
"""
Caching and Optimization System for Recipe Processing
Multi-level caching with Redis, database, and file system layers.
Implements intelligent cache management, precomputation, and optimization strategies.
"""

import os
import sys
import json
import hashlib
import pickle
import gzip
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from functools import wraps
import time

# Caching and storage
import redis
import aioredis
from redis.exceptions import RedisError
import memcache
from sqlalchemy.orm import Session

# Image processing for hash generation
import cv2
import numpy as np
from PIL import Image
import imagehash

# Monitoring
import structlog
from prometheus_client import Counter, Histogram, Gauge

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import database models
from api.database_models import ProcessingCache, RecipeProcessingJob, ExtractedIngredient, get_db_session

# Metrics
CACHE_OPERATIONS = Counter('cache_operations_total', 'Total cache operations', ['operation', 'cache_type', 'result'])
CACHE_HIT_RATIO = Gauge('cache_hit_ratio', 'Cache hit ratio by type', ['cache_type'])
CACHE_LATENCY = Histogram('cache_operation_duration_seconds', 'Cache operation latency', ['operation', 'cache_type'])
CACHE_SIZE = Gauge('cache_size_bytes', 'Cache size in bytes', ['cache_type'])
CACHE_EVICTIONS = Counter('cache_evictions_total', 'Total cache evictions', ['cache_type', 'reason'])

# Configuration
class CacheConfig:
    """Configuration for caching system."""
    
    # Redis configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
    
    # Memcache configuration
    MEMCACHE_SERVERS = os.getenv("MEMCACHE_SERVERS", "localhost:11211").split(",")
    
    # File system cache
    CACHE_DIR = Path(os.getenv("CACHE_DIR", "/tmp/recipe_cache"))
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "10737418240"))  # 10GB
    
    # TTL settings (in seconds)
    DEFAULT_TTL = int(os.getenv("DEFAULT_CACHE_TTL", "3600"))  # 1 hour
    PROCESSING_RESULT_TTL = int(os.getenv("PROCESSING_RESULT_TTL", "86400"))  # 24 hours
    FORMAT_ANALYSIS_TTL = int(os.getenv("FORMAT_ANALYSIS_TTL", "7200"))  # 2 hours
    LAYOUT_ANALYSIS_TTL = int(os.getenv("LAYOUT_ANALYSIS_TTL", "7200"))  # 2 hours
    IMAGE_HASH_TTL = int(os.getenv("IMAGE_HASH_TTL", "604800"))  # 1 week
    
    # Optimization settings
    PRECOMPUTE_POPULAR_FORMATS = bool(os.getenv("PRECOMPUTE_POPULAR_FORMATS", "true").lower() == "true")
    ENABLE_COMPRESSION = bool(os.getenv("ENABLE_COMPRESSION", "true").lower() == "true")
    COMPRESSION_LEVEL = int(os.getenv("COMPRESSION_LEVEL", "6"))
    
    # Performance settings
    BATCH_SIZE = int(os.getenv("CACHE_BATCH_SIZE", "100"))
    MAX_CONCURRENT_OPERATIONS = int(os.getenv("MAX_CONCURRENT_OPERATIONS", "50"))
    
    # Cleanup settings
    CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", "3600"))  # 1 hour
    MAX_CACHE_AGE = int(os.getenv("MAX_CACHE_AGE", "2592000"))  # 30 days

config = CacheConfig()

# Ensure cache directory exists
config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logger = structlog.get_logger(__name__)

# Enums
class CacheType(Enum):
    REDIS = "redis"
    MEMCACHE = "memcache"
    DATABASE = "database"
    FILESYSTEM = "filesystem"

class CacheResult(Enum):
    HIT = "hit"
    MISS = "miss"
    ERROR = "error"

class CacheOperation(Enum):
    GET = "get"
    SET = "set"
    DELETE = "delete"
    EVICT = "evict"

# Data Classes
@dataclass
class CacheKey:
    """Structured cache key."""
    prefix: str
    identifier: str
    version: str = "v1"
    
    def __str__(self) -> str:
        return f"{self.prefix}:{self.version}:{self.identifier}"

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = None

@dataclass
class CacheStats:
    """Cache statistics."""
    hit_count: int = 0
    miss_count: int = 0
    error_count: int = 0
    eviction_count: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_ratio(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

# Cache Interface
class CacheInterface:
    """Abstract cache interface."""
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        raise NotImplementedError
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        raise NotImplementedError
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        raise NotImplementedError
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        raise NotImplementedError
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        raise NotImplementedError

# Redis Cache Implementation
class RedisCache(CacheInterface):
    """Redis-based cache implementation."""
    
    def __init__(self):
        self.redis_client = None
        self.stats = CacheStats()
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = aioredis.from_url(
                config.REDIS_URL,
                password=config.REDIS_PASSWORD,
                db=config.REDIS_DB,
                max_connections=config.REDIS_MAX_CONNECTIONS,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            with CACHE_LATENCY.labels(operation="get", cache_type="redis").time():
                data = await self.redis_client.get(key)
                
                if data:
                    self.stats.hit_count += 1
                    CACHE_OPERATIONS.labels(operation="get", cache_type="redis", result="hit").inc()
                    
                    if config.ENABLE_COMPRESSION:
                        data = gzip.decompress(data.encode()).decode()
                    
                    return json.loads(data)
                else:
                    self.stats.miss_count += 1
                    CACHE_OPERATIONS.labels(operation="get", cache_type="redis", result="miss").inc()
                    return None
                    
        except Exception as e:
            self.stats.error_count += 1
            CACHE_OPERATIONS.labels(operation="get", cache_type="redis", result="error").inc()
            logger.error(f"Redis cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        try:
            with CACHE_LATENCY.labels(operation="set", cache_type="redis").time():
                data = json.dumps(value)
                
                if config.ENABLE_COMPRESSION:
                    data = gzip.compress(data.encode()).decode()
                
                ttl = ttl or config.DEFAULT_TTL
                await self.redis_client.setex(key, ttl, data)
                
                CACHE_OPERATIONS.labels(operation="set", cache_type="redis", result="success").inc()
                return True
                
        except Exception as e:
            CACHE_OPERATIONS.labels(operation="set", cache_type="redis", result="error").inc()
            logger.error(f"Redis cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            with CACHE_LATENCY.labels(operation="delete", cache_type="redis").time():
                result = await self.redis_client.delete(key)
                CACHE_OPERATIONS.labels(operation="delete", cache_type="redis", result="success").inc()
                return result > 0
                
        except Exception as e:
            CACHE_OPERATIONS.labels(operation="delete", cache_type="redis", result="error").inc()
            logger.error(f"Redis cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis cache exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all Redis cache entries."""
        try:
            await self.redis_client.flushdb()
            CACHE_OPERATIONS.labels(operation="clear", cache_type="redis", result="success").inc()
            return True
        except Exception as e:
            CACHE_OPERATIONS.labels(operation="clear", cache_type="redis", result="error").inc()
            logger.error(f"Redis cache clear error: {e}")
            return False
    
    async def get_stats(self) -> CacheStats:
        """Get Redis cache statistics."""
        try:
            info = await self.redis_client.info()
            self.stats.total_size_bytes = info.get('used_memory', 0)
            self.stats.entry_count = info.get('db0', {}).get('keys', 0)
            return self.stats
        except Exception as e:
            logger.error(f"Redis cache stats error: {e}")
            return self.stats

# Database Cache Implementation
class DatabaseCache(CacheInterface):
    """Database-based cache implementation."""
    
    def __init__(self):
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from database cache."""
        try:
            with CACHE_LATENCY.labels(operation="get", cache_type="database").time():
                with next(get_db_session()) as session:
                    cache_entry = session.query(ProcessingCache).filter(
                        ProcessingCache.cache_key == key
                    ).first()
                    
                    if cache_entry and not cache_entry.is_expired:
                        # Update access info
                        cache_entry.update_hit_count()
                        session.commit()
                        
                        self.stats.hit_count += 1
                        CACHE_OPERATIONS.labels(operation="get", cache_type="database", result="hit").inc()
                        
                        return cache_entry.cache_data
                    else:
                        self.stats.miss_count += 1
                        CACHE_OPERATIONS.labels(operation="get", cache_type="database", result="miss").inc()
                        return None
                        
        except Exception as e:
            self.stats.error_count += 1
            CACHE_OPERATIONS.labels(operation="get", cache_type="database", result="error").inc()
            logger.error(f"Database cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in database cache."""
        try:
            with CACHE_LATENCY.labels(operation="set", cache_type="database").time():
                with next(get_db_session()) as session:
                    # Calculate file hash if dealing with results
                    file_hash = self._calculate_result_hash(value)
                    
                    # Calculate expiration
                    ttl = ttl or config.DEFAULT_TTL
                    expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                    
                    # Check if entry exists
                    cache_entry = session.query(ProcessingCache).filter(
                        ProcessingCache.cache_key == key
                    ).first()
                    
                    if cache_entry:
                        # Update existing entry
                        cache_entry.cache_data = value
                        cache_entry.expires_at = expires_at
                        cache_entry.updated_at = datetime.utcnow()
                    else:
                        # Create new entry
                        cache_entry = ProcessingCache(
                            cache_key=key,
                            file_hash=file_hash,
                            cache_data=value,
                            expires_at=expires_at
                        )
                        session.add(cache_entry)
                    
                    session.commit()
                    CACHE_OPERATIONS.labels(operation="set", cache_type="database", result="success").inc()
                    return True
                    
        except Exception as e:
            CACHE_OPERATIONS.labels(operation="set", cache_type="database", result="error").inc()
            logger.error(f"Database cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from database cache."""
        try:
            with CACHE_LATENCY.labels(operation="delete", cache_type="database").time():
                with next(get_db_session()) as session:
                    result = session.query(ProcessingCache).filter(
                        ProcessingCache.cache_key == key
                    ).delete()
                    session.commit()
                    
                    CACHE_OPERATIONS.labels(operation="delete", cache_type="database", result="success").inc()
                    return result > 0
                    
        except Exception as e:
            CACHE_OPERATIONS.labels(operation="delete", cache_type="database", result="error").inc()
            logger.error(f"Database cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in database cache."""
        try:
            with next(get_db_session()) as session:
                return session.query(ProcessingCache).filter(
                    ProcessingCache.cache_key == key
                ).first() is not None
        except Exception as e:
            logger.error(f"Database cache exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all database cache entries."""
        try:
            with next(get_db_session()) as session:
                session.query(ProcessingCache).delete()
                session.commit()
                CACHE_OPERATIONS.labels(operation="clear", cache_type="database", result="success").inc()
                return True
        except Exception as e:
            CACHE_OPERATIONS.labels(operation="clear", cache_type="database", result="error").inc()
            logger.error(f"Database cache clear error: {e}")
            return False
    
    async def get_stats(self) -> CacheStats:
        """Get database cache statistics."""
        try:
            with next(get_db_session()) as session:
                self.stats.entry_count = session.query(ProcessingCache).count()
                return self.stats
        except Exception as e:
            logger.error(f"Database cache stats error: {e}")
            return self.stats
    
    def _calculate_result_hash(self, value: Any) -> str:
        """Calculate hash for cache value."""
        try:
            data_str = json.dumps(value, sort_keys=True)
            return hashlib.sha256(data_str.encode()).hexdigest()
        except Exception:
            return hashlib.sha256(str(value).encode()).hexdigest()

# File System Cache Implementation
class FilesystemCache(CacheInterface):
    """File system-based cache implementation."""
    
    def __init__(self):
        self.cache_dir = config.CACHE_DIR
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from filesystem cache."""
        try:
            with CACHE_LATENCY.labels(operation="get", cache_type="filesystem").time():
                cache_file = self._get_cache_file(key)
                
                if cache_file.exists():
                    # Check if expired
                    if self._is_expired(cache_file):
                        await self.delete(key)
                        self.stats.miss_count += 1
                        CACHE_OPERATIONS.labels(operation="get", cache_type="filesystem", result="miss").inc()
                        return None
                    
                    # Read and return data
                    with open(cache_file, 'rb') as f:
                        if config.ENABLE_COMPRESSION:
                            data = pickle.loads(gzip.decompress(f.read()))
                        else:
                            data = pickle.load(f)
                    
                    self.stats.hit_count += 1
                    CACHE_OPERATIONS.labels(operation="get", cache_type="filesystem", result="hit").inc()
                    return data
                else:
                    self.stats.miss_count += 1
                    CACHE_OPERATIONS.labels(operation="get", cache_type="filesystem", result="miss").inc()
                    return None
                    
        except Exception as e:
            self.stats.error_count += 1
            CACHE_OPERATIONS.labels(operation="get", cache_type="filesystem", result="error").inc()
            logger.error(f"Filesystem cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in filesystem cache."""
        try:
            with CACHE_LATENCY.labels(operation="set", cache_type="filesystem").time():
                cache_file = self._get_cache_file(key)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Prepare data with metadata
                ttl = ttl or config.DEFAULT_TTL
                cache_data = {
                    'value': value,
                    'created_at': datetime.utcnow(),
                    'expires_at': datetime.utcnow() + timedelta(seconds=ttl)
                }
                
                # Write to file
                with open(cache_file, 'wb') as f:
                    if config.ENABLE_COMPRESSION:
                        f.write(gzip.compress(pickle.dumps(cache_data), compresslevel=config.COMPRESSION_LEVEL))
                    else:
                        pickle.dump(cache_data, f)
                
                CACHE_OPERATIONS.labels(operation="set", cache_type="filesystem", result="success").inc()
                return True
                
        except Exception as e:
            CACHE_OPERATIONS.labels(operation="set", cache_type="filesystem", result="error").inc()
            logger.error(f"Filesystem cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from filesystem cache."""
        try:
            with CACHE_LATENCY.labels(operation="delete", cache_type="filesystem").time():
                cache_file = self._get_cache_file(key)
                
                if cache_file.exists():
                    cache_file.unlink()
                    CACHE_OPERATIONS.labels(operation="delete", cache_type="filesystem", result="success").inc()
                    return True
                else:
                    return False
                    
        except Exception as e:
            CACHE_OPERATIONS.labels(operation="delete", cache_type="filesystem", result="error").inc()
            logger.error(f"Filesystem cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in filesystem cache."""
        try:
            cache_file = self._get_cache_file(key)
            return cache_file.exists() and not self._is_expired(cache_file)
        except Exception as e:
            logger.error(f"Filesystem cache exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all filesystem cache entries."""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            CACHE_OPERATIONS.labels(operation="clear", cache_type="filesystem", result="success").inc()
            return True
        except Exception as e:
            CACHE_OPERATIONS.labels(operation="clear", cache_type="filesystem", result="error").inc()
            logger.error(f"Filesystem cache clear error: {e}")
            return False
    
    async def get_stats(self) -> CacheStats:
        """Get filesystem cache statistics."""
        try:
            total_size = 0
            entry_count = 0
            
            for cache_file in self.cache_dir.rglob("*.cache"):
                if cache_file.is_file():
                    total_size += cache_file.stat().st_size
                    entry_count += 1
            
            self.stats.total_size_bytes = total_size
            self.stats.entry_count = entry_count
            return self.stats
        except Exception as e:
            logger.error(f"Filesystem cache stats error: {e}")
            return self.stats
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        # Create subdirectories based on key hash to avoid too many files in one directory
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        subdir = key_hash[:2]
        return self.cache_dir / subdir / f"{key_hash}.cache"
    
    def _is_expired(self, cache_file: Path) -> bool:
        """Check if cache file is expired."""
        try:
            with open(cache_file, 'rb') as f:
                if config.ENABLE_COMPRESSION:
                    data = pickle.loads(gzip.decompress(f.read()))
                else:
                    data = pickle.load(f)
            
            expires_at = data.get('expires_at')
            if expires_at and expires_at < datetime.utcnow():
                return True
            
            return False
        except Exception:
            return True  # Consider corrupted files as expired

# Multi-Level Cache Manager
class MultiLevelCacheManager:
    """Multi-level cache manager with intelligent routing."""
    
    def __init__(self):
        self.redis_cache = RedisCache()
        self.database_cache = DatabaseCache()
        self.filesystem_cache = FilesystemCache()
        self.cache_levels = []
    
    async def initialize(self):
        """Initialize all cache levels."""
        try:
            # Initialize Redis cache
            await self.redis_cache.initialize()
            self.cache_levels.append(self.redis_cache)
            
            # Add database cache
            self.cache_levels.append(self.database_cache)
            
            # Add filesystem cache
            self.cache_levels.append(self.filesystem_cache)
            
            logger.info("Multi-level cache manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with multi-level lookup."""
        for i, cache in enumerate(self.cache_levels):
            try:
                value = await cache.get(key)
                if value is not None:
                    # Promote to higher cache levels
                    await self._promote_to_higher_levels(key, value, i)
                    return value
            except Exception as e:
                logger.error(f"Error getting from cache level {i}: {e}")
                continue
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in all cache levels."""
        success_count = 0
        
        for cache in self.cache_levels:
            try:
                if await cache.set(key, value, ttl):
                    success_count += 1
            except Exception as e:
                logger.error(f"Error setting in cache: {e}")
                continue
        
        return success_count > 0
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels."""
        success_count = 0
        
        for cache in self.cache_levels:
            try:
                if await cache.delete(key):
                    success_count += 1
            except Exception as e:
                logger.error(f"Error deleting from cache: {e}")
                continue
        
        return success_count > 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in any cache level."""
        for cache in self.cache_levels:
            try:
                if await cache.exists(key):
                    return True
            except Exception as e:
                logger.error(f"Error checking cache existence: {e}")
                continue
        
        return False
    
    async def clear(self) -> bool:
        """Clear all cache levels."""
        success_count = 0
        
        for cache in self.cache_levels:
            try:
                if await cache.clear():
                    success_count += 1
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                continue
        
        return success_count > 0
    
    async def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all cache levels."""
        stats = {}
        
        for i, cache in enumerate(self.cache_levels):
            try:
                cache_type = type(cache).__name__
                stats[cache_type] = await cache.get_stats()
            except Exception as e:
                logger.error(f"Error getting stats from cache level {i}: {e}")
                continue
        
        return stats
    
    async def _promote_to_higher_levels(self, key: str, value: Any, found_level: int):
        """Promote cache entry to higher levels."""
        for i in range(found_level):
            try:
                await self.cache_levels[i].set(key, value)
            except Exception as e:
                logger.error(f"Error promoting to cache level {i}: {e}")
                continue

# Specialized Cache Classes
class ProcessingResultCache:
    """Cache for processing results with intelligent invalidation."""
    
    def __init__(self, cache_manager: MultiLevelCacheManager):
        self.cache_manager = cache_manager
    
    async def get_processing_result(self, file_hash: str, processing_options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached processing result."""
        cache_key = self._build_processing_key(file_hash, processing_options)
        return await self.cache_manager.get(cache_key)
    
    async def set_processing_result(self, file_hash: str, processing_options: Dict[str, Any], 
                                  result: Dict[str, Any]) -> bool:
        """Set processing result in cache."""
        cache_key = self._build_processing_key(file_hash, processing_options)
        return await self.cache_manager.set(cache_key, result, config.PROCESSING_RESULT_TTL)
    
    async def get_format_analysis(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached format analysis."""
        cache_key = f"format_analysis:{file_hash}"
        return await self.cache_manager.get(cache_key)
    
    async def set_format_analysis(self, file_hash: str, analysis: Dict[str, Any]) -> bool:
        """Set format analysis in cache."""
        cache_key = f"format_analysis:{file_hash}"
        return await self.cache_manager.set(cache_key, analysis, config.FORMAT_ANALYSIS_TTL)
    
    async def get_layout_analysis(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached layout analysis."""
        cache_key = f"layout_analysis:{file_hash}"
        return await self.cache_manager.get(cache_key)
    
    async def set_layout_analysis(self, file_hash: str, analysis: Dict[str, Any]) -> bool:
        """Set layout analysis in cache."""
        cache_key = f"layout_analysis:{file_hash}"
        return await self.cache_manager.set(cache_key, analysis, config.LAYOUT_ANALYSIS_TTL)
    
    def _build_processing_key(self, file_hash: str, processing_options: Dict[str, Any]) -> str:
        """Build cache key for processing result."""
        options_str = json.dumps(processing_options, sort_keys=True)
        options_hash = hashlib.sha256(options_str.encode()).hexdigest()[:16]
        return f"processing_result:{file_hash}:{options_hash}"

class ImageHashCache:
    """Cache for image hashes and similarity detection."""
    
    def __init__(self, cache_manager: MultiLevelCacheManager):
        self.cache_manager = cache_manager
    
    async def get_image_hash(self, file_path: str) -> Optional[str]:
        """Get cached image hash."""
        cache_key = f"image_hash:{file_path}"
        return await self.cache_manager.get(cache_key)
    
    async def set_image_hash(self, file_path: str, image_hash: str) -> bool:
        """Set image hash in cache."""
        cache_key = f"image_hash:{file_path}"
        return await self.cache_manager.set(cache_key, image_hash, config.IMAGE_HASH_TTL)
    
    async def find_similar_images(self, target_hash: str, threshold: int = 5) -> List[str]:
        """Find similar images based on hash."""
        # This would require implementing similarity search
        # For now, return empty list
        return []
    
    def calculate_perceptual_hash(self, image_path: str) -> str:
        """Calculate perceptual hash for image."""
        try:
            image = Image.open(image_path)
            return str(imagehash.phash(image))
        except Exception as e:
            logger.error(f"Error calculating perceptual hash: {e}")
            # Fallback to file hash
            return self.calculate_file_hash(image_path)
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return hashlib.sha256(file_path.encode()).hexdigest()

# Cache Decorators
def cache_result(ttl: Optional[int] = None, key_prefix: str = "cache"):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            key_data = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            cache_key = f"{key_prefix}:{hashlib.sha256(key_data.encode()).hexdigest()}"
            
            # Try to get from cache
            if hasattr(wrapper, '_cache_manager'):
                cached_result = await wrapper._cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if hasattr(wrapper, '_cache_manager') and result is not None:
                await wrapper._cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

# Cache Cleanup and Maintenance
class CacheCleanup:
    """Cache cleanup and maintenance utilities."""
    
    def __init__(self, cache_manager: MultiLevelCacheManager):
        self.cache_manager = cache_manager
    
    async def cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        try:
            # Database cleanup
            with next(get_db_session()) as session:
                expired_count = session.query(ProcessingCache).filter(
                    ProcessingCache.expires_at < datetime.utcnow()
                ).delete()
                session.commit()
                
                logger.info(f"Cleaned up {expired_count} expired database cache entries")
            
            # Filesystem cleanup
            await self._cleanup_filesystem_cache()
            
            CACHE_OPERATIONS.labels(operation="cleanup", cache_type="all", result="success").inc()
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            CACHE_OPERATIONS.labels(operation="cleanup", cache_type="all", result="error").inc()
    
    async def _cleanup_filesystem_cache(self):
        """Clean up expired filesystem cache entries."""
        try:
            cleanup_count = 0
            for cache_file in config.CACHE_DIR.rglob("*.cache"):
                if cache_file.is_file():
                    try:
                        # Check if expired
                        if self._is_filesystem_file_expired(cache_file):
                            cache_file.unlink()
                            cleanup_count += 1
                    except Exception as e:
                        logger.error(f"Error cleaning up cache file {cache_file}: {e}")
            
            logger.info(f"Cleaned up {cleanup_count} expired filesystem cache entries")
            
        except Exception as e:
            logger.error(f"Filesystem cache cleanup error: {e}")
    
    def _is_filesystem_file_expired(self, cache_file: Path) -> bool:
        """Check if filesystem cache file is expired."""
        try:
            with open(cache_file, 'rb') as f:
                if config.ENABLE_COMPRESSION:
                    data = pickle.loads(gzip.decompress(f.read()))
                else:
                    data = pickle.load(f)
            
            expires_at = data.get('expires_at')
            return expires_at and expires_at < datetime.utcnow()
        except Exception:
            return True  # Consider corrupted files as expired

# Global cache manager instance
cache_manager = MultiLevelCacheManager()
processing_cache = ProcessingResultCache(cache_manager)
image_cache = ImageHashCache(cache_manager)
cache_cleanup = CacheCleanup(cache_manager)

# Initialization function
async def initialize_cache_system():
    """Initialize the cache system."""
    await cache_manager.initialize()
    logger.info("Cache system initialized successfully")

# Cleanup task
async def run_cache_cleanup():
    """Run periodic cache cleanup."""
    while True:
        try:
            await cache_cleanup.cleanup_expired_entries()
            await asyncio.sleep(config.CLEANUP_INTERVAL)
        except Exception as e:
            logger.error(f"Cache cleanup task error: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes before retry

if __name__ == "__main__":
    # Test cache system
    async def test_cache():
        await initialize_cache_system()
        
        # Test basic operations
        await cache_manager.set("test_key", {"data": "test_value"})
        result = await cache_manager.get("test_key")
        print(f"Cache test result: {result}")
        
        # Test processing cache
        await processing_cache.set_processing_result("test_hash", {"option": "value"}, {"result": "test"})
        cached_result = await processing_cache.get_processing_result("test_hash", {"option": "value"})
        print(f"Processing cache test result: {cached_result}")
        
        # Get stats
        stats = await cache_manager.get_stats()
        print(f"Cache stats: {stats}")
    
    asyncio.run(test_cache())