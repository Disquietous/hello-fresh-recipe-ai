# Recipe Processing API Documentation

## Overview

The Recipe Processing API provides comprehensive text extraction and ingredient recognition from recipe images. It supports multiple formats, languages, and measurement systems with production-grade features including batch processing, caching, and monitoring.

## Base URL

- Production: `https://api.recipe-processing.com`
- Staging: `https://staging-api.recipe-processing.com`
- Development: `http://localhost:8000`

## Authentication

The API uses Bearer token authentication. Include your API key in the Authorization header:

```
Authorization: Bearer YOUR_API_KEY
```

## Rate Limits

- **Free Tier**: 100 requests/hour, 1,000 requests/day
- **Pro Tier**: 1,000 requests/hour, 10,000 requests/day
- **Enterprise**: Custom limits

## API Endpoints

### Health Check

#### GET /health

Check the overall health status of the API and its dependencies.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "message": "Database connection successful",
      "response_time": 0.025,
      "details": {
        "response_time": 0.025,
        "url": "postgres:5432"
      }
    },
    "redis": {
      "status": "healthy",
      "message": "Redis connection successful",
      "response_time": 0.012,
      "details": {
        "response_time": 0.012,
        "used_memory": "15.2MB",
        "connected_clients": 3
      }
    },
    "file_system": {
      "status": "healthy",
      "message": "File system healthy",
      "response_time": 0.005,
      "details": {
        "free_space_percent": 75.3,
        "temp_directory_writable": true,
        "total_space": 1073741824,
        "free_space": 807403520
      }
    }
  }
}
```

**Status Codes:**
- `200 OK`: All systems healthy
- `503 Service Unavailable`: One or more systems unhealthy

---

### Process Single Recipe

#### POST /process

Process a single recipe image and extract ingredients.

**Request:**

**Headers:**
- `Content-Type: multipart/form-data`
- `Authorization: Bearer YOUR_API_KEY`

**Form Data:**
- `file` (required): Recipe image file (JPG, PNG, WEBP, etc.)
- `format_hint` (optional): Format hint (`"cookbook"`, `"handwritten"`, `"digital"`, `"blog"`)
- `language_hint` (optional): Language hint (`"en"`, `"es"`, `"fr"`, `"de"`, etc.)
- `measurement_system_hint` (optional): Measurement system (`"metric"`, `"imperial"`, `"mixed"`)
- `quality_threshold` (optional): Minimum quality threshold (0.0-1.0, default: 0.5)
- `confidence_threshold` (optional): Minimum confidence threshold (0.0-1.0, default: 0.25)
- `enable_caching` (optional): Enable result caching (default: true)

**cURL Example:**
```bash
curl -X POST "https://api.recipe-processing.com/process" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@recipe.jpg" \
  -F "format_hint=cookbook" \
  -F "language_hint=en" \
  -F "measurement_system_hint=imperial" \
  -F "quality_threshold=0.6"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "filename": "recipe.jpg",
  "processing_time": 15.42,
  "format_type": "printed_cookbook",
  "language": "en",
  "measurement_system": "imperial",
  "quality_score": 0.85,
  "confidence_score": 0.78,
  "ingredients": [
    {
      "ingredient_name": "flour",
      "ingredient_name_en": "flour",
      "quantity": "2",
      "unit": "cups",
      "unit_normalized": "cup",
      "preparation": "sifted",
      "confidence": 0.92,
      "bbox": {
        "x1": 120,
        "y1": 45,
        "x2": 280,
        "y2": 70
      }
    },
    {
      "ingredient_name": "sugar",
      "ingredient_name_en": "sugar",
      "quantity": "1",
      "unit": "cup",
      "unit_normalized": "cup",
      "preparation": null,
      "confidence": 0.89,
      "bbox": {
        "x1": 120,
        "y1": 75,
        "x2": 250,
        "y2": 100
      }
    }
  ],
  "metadata": {
    "format_analysis": {
      "detected_format": "printed_cookbook",
      "confidence": 0.85,
      "quality_score": 0.85,
      "layout_type": "single_column",
      "characteristics": {
        "image_size": [1024, 768],
        "aspect_ratio": 1.33,
        "brightness_mean": 180.5,
        "text_regions_count": 8
      }
    },
    "language_result": {
      "primary_language": "en",
      "confidence": 0.95,
      "script_type": "latin",
      "text_direction": "ltr"
    }
  },
  "created_at": "2024-01-15T10:25:00Z",
  "completed_at": "2024-01-15T10:25:15Z"
}
```

**Status Codes:**
- `200 OK`: Processing successful
- `400 Bad Request`: Invalid request parameters
- `413 Payload Too Large`: File size exceeds limit
- `422 Unprocessable Entity`: Invalid file format
- `500 Internal Server Error`: Processing failed

---

### Batch Processing

#### POST /batch

Start batch processing of multiple recipe images.

**Request:**

**Headers:**
- `Content-Type: multipart/form-data`
- `Authorization: Bearer YOUR_API_KEY`

**Form Data:**
- `files` (required): Multiple recipe image files
- `job_name` (required): Name for the batch job
- `processing_options` (optional): JSON object with processing options
- `max_concurrent` (optional): Maximum concurrent jobs (1-20, default: 5)
- `priority` (optional): Processing priority (`"low"`, `"normal"`, `"high"`)

**cURL Example:**
```bash
curl -X POST "https://api.recipe-processing.com/batch" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "files=@recipe1.jpg" \
  -F "files=@recipe2.jpg" \
  -F "files=@recipe3.jpg" \
  -F "job_name=cookbook_batch_1" \
  -F "max_concurrent=10" \
  -F "priority=high"
```

**Response:**
```json
{
  "batch_id": "750f9511-f39c-42e5-b827-556766551111",
  "job_name": "cookbook_batch_1",
  "status": "processing",
  "total_jobs": 3,
  "completed_jobs": 0,
  "failed_jobs": 0,
  "progress": 0.0,
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:35:00Z"
}
```

---

#### GET /batch/{batch_id}

Get status of a batch processing job.

**Response:**
```json
{
  "batch_id": "750f9511-f39c-42e5-b827-556766551111",
  "job_name": "cookbook_batch_1",
  "status": "completed",
  "total_jobs": 3,
  "completed_jobs": 3,
  "failed_jobs": 0,
  "progress": 100.0,
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:35:00Z"
}
```

---

### Job Management

#### GET /job/{job_id}

Get details of a specific processing job.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "filename": "recipe.jpg",
  "processing_time": 15.42,
  "format_type": "printed_cookbook",
  "language": "en",
  "measurement_system": "imperial",
  "quality_score": 0.85,
  "confidence_score": 0.78,
  "ingredients": [...],
  "metadata": {...},
  "created_at": "2024-01-15T10:25:00Z",
  "completed_at": "2024-01-15T10:25:15Z"
}
```

---

#### GET /jobs

List processing jobs with pagination and filtering.

**Query Parameters:**
- `skip` (optional): Number of records to skip (default: 0)
- `limit` (optional): Maximum number of records (1-1000, default: 100)
- `status` (optional): Filter by status (`"pending"`, `"processing"`, `"completed"`, `"failed"`)

**Response:**
```json
[
  {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "filename": "recipe1.jpg",
    "processing_time": 15.42,
    "format_type": "printed_cookbook",
    "language": "en",
    "quality_score": 0.85,
    "ingredients": [],
    "metadata": {},
    "created_at": "2024-01-15T10:25:00Z",
    "completed_at": "2024-01-15T10:25:15Z"
  }
]
```

---

### Cache Management

#### DELETE /cache

Clear processing cache (requires admin privileges).

**Response:**
```json
{
  "message": "Cache cleared successfully"
}
```

---

### Metrics

#### GET /metrics

Get Prometheus metrics (for monitoring systems).

**Response:** Plain text Prometheus format
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",endpoint="/process",status_code="200"} 1523
...
```

---

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation failed",
    "severity": "medium",
    "category": "validation",
    "trace_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2024-01-15T10:30:00Z",
    "details": {
      "validation_errors": [
        "File size exceeds maximum limit of 10MB",
        "Unsupported file format: .txt"
      ]
    }
  }
}
```

### Common Error Codes

- `VALIDATION_ERROR`: Invalid request parameters
- `FILE_TOO_LARGE`: File exceeds size limit
- `UNSUPPORTED_FORMAT`: File format not supported
- `PROCESSING_FAILED`: Image processing failed
- `OCR_ERROR`: OCR extraction failed
- `TIMEOUT_ERROR`: Operation timed out
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_SERVER_ERROR`: Unexpected server error

---

## Supported Formats

### Image Formats
- JPEG/JPG
- PNG
- WEBP
- BMP
- TIFF

### Recipe Formats
- **Printed cookbook pages**: Traditional cookbook layouts
- **Handwritten recipe cards**: Personal recipe notes
- **Digital recipe screenshots**: Screenshots from apps/websites
- **Recipe blog images**: Blog-style recipe layouts
- **Mixed content**: Recipes with embedded images

### Languages Supported
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)
- Arabic (ar)
- Hindi (hi)
- Dutch (nl)
- Swedish (sv)
- Norwegian (no)
- Danish (da)
- Finnish (fi)
- Polish (pl)
- Turkish (tr)
- Greek (el)

### Measurement Systems
- **Metric**: Grams, kilograms, milliliters, liters
- **Imperial**: Ounces, pounds, cups, tablespoons, teaspoons
- **Mixed**: Combination of metric and imperial
- **Traditional Asian**: Traditional units (go, sho, etc.)
- **Traditional European**: Traditional European units

---

## SDKs and Libraries

### Python SDK

```python
import requests

class RecipeProcessingClient:
    def __init__(self, api_key, base_url="https://api.recipe-processing.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def process_image(self, image_path, **options):
        """Process a single recipe image."""
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.base_url}/process",
                headers=self.headers,
                files=files,
                data=options
            )
        return response.json()
    
    def start_batch(self, image_paths, job_name, **options):
        """Start batch processing."""
        files = [('files', open(path, 'rb')) for path in image_paths]
        try:
            response = requests.post(
                f"{self.base_url}/batch",
                headers=self.headers,
                files=files,
                data={'job_name': job_name, **options}
            )
            return response.json()
        finally:
            for _, f in files:
                f.close()
    
    def get_job_status(self, job_id):
        """Get job status."""
        response = requests.get(
            f"{self.base_url}/job/{job_id}",
            headers=self.headers
        )
        return response.json()

# Usage
client = RecipeProcessingClient("your_api_key")
result = client.process_image("recipe.jpg", format_hint="cookbook")
print(result)
```

### JavaScript SDK

```javascript
class RecipeProcessingClient {
    constructor(apiKey, baseUrl = 'https://api.recipe-processing.com') {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${apiKey}`
        };
    }

    async processImage(file, options = {}) {
        const formData = new FormData();
        formData.append('file', file);
        
        Object.keys(options).forEach(key => {
            formData.append(key, options[key]);
        });

        const response = await fetch(`${this.baseUrl}/process`, {
            method: 'POST',
            headers: this.headers,
            body: formData
        });

        return await response.json();
    }

    async startBatch(files, jobName, options = {}) {
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });
        formData.append('job_name', jobName);
        
        Object.keys(options).forEach(key => {
            formData.append(key, options[key]);
        });

        const response = await fetch(`${this.baseUrl}/batch`, {
            method: 'POST',
            headers: this.headers,
            body: formData
        });

        return await response.json();
    }

    async getJobStatus(jobId) {
        const response = await fetch(`${this.baseUrl}/job/${jobId}`, {
            headers: this.headers
        });

        return await response.json();
    }
}

// Usage
const client = new RecipeProcessingClient('your_api_key');
const result = await client.processImage(fileInput.files[0], {
    format_hint: 'cookbook',
    language_hint: 'en'
});
console.log(result);
```

---

## Webhooks

Configure webhooks to receive notifications when processing jobs complete:

### Webhook Configuration

POST /webhooks/configure

```json
{
  "url": "https://your-app.com/webhook",
  "events": ["job.completed", "job.failed", "batch.completed"],
  "secret": "webhook_secret_key"
}
```

### Webhook Payload

```json
{
  "event": "job.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "processing_time": 15.42,
    "ingredients_count": 8,
    "quality_score": 0.85
  },
  "signature": "sha256=..."
}
```

---

## Best Practices

### Image Quality
- Use high-resolution images (300+ DPI)
- Ensure good lighting and contrast
- Avoid blurry or skewed images
- Clean backgrounds work best

### Performance Optimization
- Enable caching for repeated processing
- Use batch processing for multiple images
- Specify format hints when known
- Use appropriate quality thresholds

### Error Handling
- Always check response status codes
- Implement retry logic with exponential backoff
- Monitor rate limits
- Handle timeout errors gracefully

### Security
- Store API keys securely
- Use HTTPS in production
- Validate webhook signatures
- Implement proper authentication

---

## Rate Limiting

The API implements rate limiting based on your subscription tier:

### Headers
Every response includes rate limit headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
X-RateLimit-Retry-After: 3600
```

### Handling Rate Limits
When you exceed rate limits, you'll receive a `429 Too Many Requests` response:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 1000,
      "remaining": 0,
      "reset_time": "2024-01-15T11:00:00Z",
      "retry_after": 3600
    }
  }
}
```

---

## Changelog

### v1.2.0 (2024-01-15)
- Added batch processing capabilities
- Improved multilingual support
- Enhanced error handling
- Added webhook support

### v1.1.0 (2024-01-01)
- Added format-specific processing
- Improved OCR accuracy
- Added measurement system conversion
- Enhanced monitoring and logging

### v1.0.0 (2023-12-01)
- Initial release
- Basic ingredient extraction
- Single image processing
- English language support

---

## Support

For technical support, please:
- Check the [FAQ](https://docs.recipe-processing.com/faq)
- Visit our [GitHub repository](https://github.com/recipe-processing/api)
- Contact support at support@recipe-processing.com
- Join our [Discord community](https://discord.gg/recipe-processing)

## Terms of Service

By using this API, you agree to our [Terms of Service](https://recipe-processing.com/terms) and [Privacy Policy](https://recipe-processing.com/privacy).