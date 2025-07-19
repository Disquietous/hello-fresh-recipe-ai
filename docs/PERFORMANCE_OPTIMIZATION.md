# Performance Optimization Guide

This guide provides comprehensive information about optimizing the Recipe Processing API for production use, including benchmarking, monitoring, and tuning recommendations.

## Table of Contents

1. [Overview](#overview)
2. [Performance Benchmarking](#performance-benchmarking)
3. [Optimization Strategies](#optimization-strategies)
4. [Monitoring and Alerting](#monitoring-and-alerting)
5. [Configuration Tuning](#configuration-tuning)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Overview

The Recipe Processing API is designed for high-performance text extraction and ingredient recognition from recipe images. Performance optimization involves multiple layers:

- **Application Layer**: Code optimization, caching, and algorithmic improvements
- **Infrastructure Layer**: Resource allocation, scaling, and container optimization
- **Data Layer**: Database tuning, cache optimization, and storage efficiency
- **Network Layer**: Connection pooling, compression, and CDN usage

### Performance Targets

| Metric | Development | Staging | Production |
|--------|-------------|---------|------------|
| Average Response Time | < 10s | < 7s | < 5s |
| P95 Response Time | < 20s | < 15s | < 10s |
| P99 Response Time | < 30s | < 25s | < 20s |
| Throughput | 1 req/s | 5 req/s | 10+ req/s |
| Success Rate | > 90% | > 95% | > 99% |
| Memory Usage | < 512MB | < 1GB | < 2GB |
| CPU Usage | < 70% | < 60% | < 50% |

## Performance Benchmarking

### Running Benchmarks

The performance benchmarking system provides comprehensive testing capabilities:

#### Basic Benchmark
```bash
# Run full benchmark suite
./scripts/performance_testing.sh benchmark

# Run with custom parameters
./scripts/performance_testing.sh benchmark \
  --concurrency 1,2,4,8 \
  --duration 30,60,120 \
  --output benchmarks/results
```

#### API Load Testing
```bash
# Test API endpoints
./scripts/performance_testing.sh load-test \
  --api-url http://localhost:8000 \
  --concurrency 4 \
  --duration 60

# Stress testing
./scripts/performance_testing.sh stress-test \
  --api-url http://localhost:8000 \
  --concurrency 20
```

#### Python Benchmarking
```python
from src.performance_benchmarking import PerformanceBenchmark, BenchmarkConfig

# Configure benchmark
config = BenchmarkConfig(
    test_images_dir="data/test/images",
    output_dir="benchmarks/results",
    concurrent_requests=[1, 2, 4, 8],
    test_durations=[30, 60, 120]
)

# Run benchmark
benchmark = PerformanceBenchmark(config)
await benchmark.run_full_benchmark_suite()
```

### Benchmark Metrics

The benchmarking system measures:

- **Response Time**: Request processing duration
- **Throughput**: Requests processed per second
- **Memory Usage**: Peak and current memory consumption
- **CPU Usage**: Processor utilization
- **Success Rate**: Percentage of successful requests
- **Error Rate**: Failed request frequency
- **Concurrency Performance**: Scaling under load

### Performance Profiling

#### CPU Profiling
```bash
# Enable CPU profiling during benchmark
python -m cProfile -o profile.prof src/performance_benchmarking.py

# Analyze results
python -c "
import pstats
stats = pstats.Stats('profile.prof')
stats.sort_stats('cumulative')
stats.print_stats(20)
"
```

#### Memory Profiling
```python
import tracemalloc
from memory_profiler import profile

# Start memory tracing
tracemalloc.start()

# Your processing code here
result = pipeline.process_image('test.jpg')

# Get memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f} MB")
print(f"Peak: {peak / 1024 / 1024:.1f} MB")
```

## Optimization Strategies

### 1. Image Processing Optimization

#### Image Preprocessing
```python
# Optimize image size before processing
def optimize_image_size(image_path, max_dimension=2048):
    with Image.open(image_path) as img:
        # Calculate resize ratio
        width, height = img.size
        max_dim = max(width, height)
        
        if max_dim > max_dimension:
            ratio = max_dimension / max_dim
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        return img
```

#### Batch Processing
```python
# Process multiple images in batches
async def process_batch_optimized(image_paths, batch_size=5):
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batch_results = await asyncio.gather(*[
            process_image_async(img) for img in batch
        ])
        results.extend(batch_results)
    return results
```

### 2. OCR Engine Optimization

#### Engine Selection Strategy
```python
def select_optimal_ocr_engine(image_characteristics):
    """Select OCR engine based on image characteristics"""
    if image_characteristics['text_density'] > 0.8:
        return 'paddleocr'  # Best for dense text
    elif image_characteristics['handwritten_probability'] > 0.5:
        return 'easyocr'    # Better for handwritten text
    else:
        return 'tesseract'  # Default choice
```

#### Parallel OCR Processing
```python
import concurrent.futures

def parallel_ocr_processing(text_regions, max_workers=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(ocr_engine.extract_text, region)
            for region in text_regions
        ]
        results = [future.result() for future in futures]
    return results
```

### 3. Caching Optimization

#### Multi-Level Caching
```python
class OptimizedCacheManager:
    def __init__(self):
        self.memory_cache = {}  # L1: Memory cache
        self.redis_cache = redis.Redis()  # L2: Redis cache
        self.disk_cache = DiskCache()  # L3: Disk cache
    
    async def get_with_fallback(self, key):
        # Try memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Try Redis cache
        value = await self.redis_cache.get(key)
        if value:
            self.memory_cache[key] = value  # Promote to L1
            return value
        
        # Try disk cache
        value = await self.disk_cache.get(key)
        if value:
            await self.redis_cache.set(key, value)  # Promote to L2
            self.memory_cache[key] = value  # Promote to L1
            return value
        
        return None
```

#### Cache Warming
```python
async def warm_cache(common_images):
    """Pre-populate cache with frequently accessed images"""
    for image_path in common_images:
        if not await cache_manager.exists(image_path):
            result = await process_image(image_path)
            await cache_manager.set(image_path, result)
```

### 4. Database Optimization

#### Connection Pooling
```python
from sqlalchemy.pool import QueuePool

# Optimize database connection pool
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

#### Query Optimization
```python
# Use efficient queries with proper indexing
async def get_recent_jobs_optimized(limit=100):
    query = select(RecipeProcessingJob).where(
        RecipeProcessingJob.created_at >= datetime.now() - timedelta(days=7)
    ).order_by(
        RecipeProcessingJob.created_at.desc()
    ).limit(limit)
    
    result = await session.execute(query)
    return result.scalars().all()
```

### 5. Memory Management

#### Garbage Collection Tuning
```python
import gc

def optimize_memory_usage():
    # Force garbage collection
    gc.collect()
    
    # Adjust GC thresholds for better performance
    gc.set_threshold(700, 10, 10)
```

#### Memory-Efficient Processing
```python
def process_large_batch_memory_efficient(image_paths):
    """Process large batches without memory overflow"""
    for chunk in chunked(image_paths, chunk_size=10):
        # Process chunk
        results = process_chunk(chunk)
        
        # Yield results to prevent memory accumulation
        for result in results:
            yield result
        
        # Clear intermediate variables
        del results
        gc.collect()
```

## Monitoring and Alerting

### Performance Metrics Collection

#### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage')

# Instrument your code
@REQUEST_DURATION.time()
async def process_request():
    REQUEST_COUNT.labels(method='POST', endpoint='/process').inc()
    # Processing logic here
```

#### Custom Health Checks
```python
async def comprehensive_health_check():
    checks = {
        'database': await check_database_connection(),
        'redis': await check_redis_connection(),
        'disk_space': check_disk_space(),
        'memory_usage': check_memory_usage(),
        'cpu_usage': check_cpu_usage()
    }
    
    overall_health = all(check['status'] == 'healthy' for check in checks.values())
    return {'status': 'healthy' if overall_health else 'unhealthy', 'checks': checks}
```

### Alerting Rules

#### Performance Alerts
```yaml
# Prometheus alerting rules
groups:
  - name: recipe_processing_performance
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m])) > 30
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          
      - alert: HighErrorRate
        expr: rate(requests_total{status=~"5.."}[5m]) / rate(requests_total[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
```

### Grafana Dashboards

#### Key Performance Dashboard
```json
{
  "dashboard": {
    "title": "Recipe Processing Performance",
    "panels": [
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      }
    ]
  }
}
```

## Configuration Tuning

### Environment-Specific Configurations

#### Production Configuration
```yaml
# configs/performance_config.yaml
production:
  max_workers: 8
  max_concurrent_requests: 16
  processing_timeout: 60
  cache_ttl: 3600
  memory_limit_mb: 2048
  batch_size_limit: 50
```

#### Auto-Scaling Configuration
```yaml
autoscaling:
  triggers:
    cpu_threshold: 0.7
    memory_threshold: 0.8
    response_time_threshold: 15.0
  min_instances: 2
  max_instances: 10
  scale_up_cooldown: 300
  scale_down_cooldown: 600
```

### Load Balancer Configuration

#### NGINX Configuration
```nginx
upstream recipe_api {
    least_conn;
    server app1:8000 max_fails=3 fail_timeout=30s;
    server app2:8000 max_fails=3 fail_timeout=30s;
    server app3:8000 max_fails=3 fail_timeout=30s;
}

server {
    location / {
        proxy_pass http://recipe_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Performance optimizations
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_connect_timeout 10s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
}
```

## Troubleshooting

### Common Performance Issues

#### High Memory Usage
```bash
# Diagnose memory issues
./scripts/performance_testing.sh profile --memory-profiling

# Check for memory leaks
python -m memory_profiler src/ingredient_pipeline.py
```

#### Slow Response Times
```bash
# Profile CPU usage
./scripts/performance_testing.sh profile --cpu-profiling

# Analyze database queries
# Check slow query logs and optimize indexes
```

#### High Error Rates
```bash
# Check application logs
docker-compose logs recipe-api | grep ERROR

# Run health checks
curl http://localhost:8000/health
```

### Performance Debugging

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add performance logging
logger = logging.getLogger(__name__)
start_time = time.time()
# ... processing code ...
logger.debug(f"Processing took {time.time() - start_time:.3f} seconds")
```

#### Trace Analysis
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("image_processing"):
    with tracer.start_as_current_span("text_detection"):
        # Text detection code
        pass
    with tracer.start_as_current_span("ocr_processing"):
        # OCR processing code
        pass
```

## Best Practices

### Development Best Practices

1. **Profile Early and Often**: Regular performance testing during development
2. **Benchmark Regression Testing**: Ensure new features don't degrade performance
3. **Memory Management**: Proper cleanup of large objects and temporary files
4. **Async Processing**: Use async/await for I/O-bound operations
5. **Resource Limits**: Set appropriate limits for memory, CPU, and file sizes

### Production Best Practices

1. **Horizontal Scaling**: Scale out rather than up when possible
2. **Circuit Breakers**: Implement fault tolerance for external dependencies
3. **Rate Limiting**: Protect against abuse and resource exhaustion
4. **Health Checks**: Comprehensive monitoring of all system components
5. **Graceful Degradation**: Fallback mechanisms for high-load scenarios

### Monitoring Best Practices

1. **SLA Definition**: Clear performance targets and SLA metrics
2. **Alerting Strategy**: Actionable alerts with proper escalation
3. **Capacity Planning**: Proactive resource planning based on trends
4. **Performance Budgets**: Set and enforce performance budgets
5. **Regular Reviews**: Periodic performance review and optimization

## Additional Resources

- [Performance Configuration Reference](../configs/performance_config.yaml)
- [Benchmarking Scripts](../scripts/performance_testing.sh)
- [Monitoring Setup](./MONITORING_SETUP.md)
- [API Documentation](./API_DOCUMENTATION.md)
- [Deployment Guide](./DEPLOYMENT_GUIDE.md)

---

For questions or support, please refer to the main documentation or contact the development team.