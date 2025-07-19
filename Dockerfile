# Production Dockerfile for Recipe Processing API
# Multi-stage build for optimized container size and security

# Build stage
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt

# Production stage
FROM python:3.9-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    # Core system utilities
    curl \
    wget \
    unzip \
    # Image processing libraries
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    # OCR dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-ita \
    # Additional language packs
    libtesseract-dev \
    libleptonica-dev \
    # Fonts for better OCR
    fonts-liberation \
    fonts-dejavu-core \
    # Process monitoring
    htop \
    procps \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r recipeapp && useradd -r -g recipeapp recipeapp

# Create application directories
RUN mkdir -p /app /app/logs /app/cache /app/models /app/temp /app/data \
    && chown -R recipeapp:recipeapp /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=recipeapp:recipeapp . /app/

# Create additional directories and set permissions
RUN mkdir -p /var/log/recipe-processing \
    && chown -R recipeapp:recipeapp /var/log/recipe-processing \
    && chmod -R 755 /app \
    && chmod +x /app/scripts/*.sh 2>/dev/null || true

# Download pre-trained models (if needed)
RUN mkdir -p /app/models/pretrained \
    && chown -R recipeapp:recipeapp /app/models

# Install any additional Python packages from app requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER recipeapp

# Set Python path
ENV PYTHONPATH="/app:/app/src:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8080

# Default command
CMD ["python", "-m", "uvicorn", "src.api.recipe_processing_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]