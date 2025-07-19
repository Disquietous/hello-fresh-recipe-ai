#!/bin/bash
# Docker entrypoint script for Recipe Processing API
# Handles initialization, database migrations, and service startup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Default environment variables
export PYTHONPATH="${PYTHONPATH:-/app:/app/src}"
export ENVIRONMENT="${ENVIRONMENT:-production}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}
    
    log "Waiting for $service_name at $host:$port..."
    
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port" 2>/dev/null; then
            success "$service_name is ready!"
            return 0
        fi
        
        if [ $i -eq $timeout ]; then
            error "$service_name is not available after $timeout seconds"
            return 1
        fi
        
        sleep 1
    done
}

# Function to wait for database
wait_for_database() {
    log "Waiting for database..."
    
    # Extract database connection details from DATABASE_URL
    if [ -n "$DATABASE_URL" ]; then
        # Parse DATABASE_URL
        # Format: postgresql://user:password@host:port/database
        DB_HOST=$(echo $DATABASE_URL | sed 's/.*@\([^:]*\):.*/\1/')
        DB_PORT=$(echo $DATABASE_URL | sed 's/.*:\([0-9]*\)\/.*/\1/')
        
        if [ -n "$DB_HOST" ] && [ -n "$DB_PORT" ]; then
            wait_for_service "$DB_HOST" "$DB_PORT" "Database" 60
        else
            warning "Could not parse DATABASE_URL for connection check"
        fi
    else
        warning "DATABASE_URL not set, skipping database connection check"
    fi
}

# Function to wait for Redis
wait_for_redis() {
    log "Waiting for Redis..."
    
    # Extract Redis connection details from REDIS_URL
    if [ -n "$REDIS_URL" ]; then
        # Format: redis://host:port
        REDIS_HOST=$(echo $REDIS_URL | sed 's/redis:\/\/\([^:]*\):.*/\1/')
        REDIS_PORT=$(echo $REDIS_URL | sed 's/.*:\([0-9]*\).*/\1/')
        
        if [ -n "$REDIS_HOST" ] && [ -n "$REDIS_PORT" ]; then
            wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis" 30
        else
            warning "Could not parse REDIS_URL for connection check"
        fi
    else
        warning "REDIS_URL not set, skipping Redis connection check"
    fi
}

# Function to run database migrations
run_migrations() {
    log "Running database migrations..."
    
    python -c "
import sys
sys.path.append('/app')
sys.path.append('/app/src')

try:
    from api.database_models import init_database
    init_database()
    print('Database migrations completed successfully')
except Exception as e:
    print(f'Database migration failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        success "Database migrations completed"
    else
        error "Database migrations failed"
        exit 1
    fi
}

# Function to download models if needed
download_models() {
    log "Checking for required models..."
    
    MODEL_DIR="/app/models/pretrained"
    mkdir -p "$MODEL_DIR"
    
    # Check if YOLOv8 model exists
    if [ ! -f "$MODEL_DIR/yolov8n.pt" ]; then
        log "Downloading YOLOv8 model..."
        python -c "
import sys
sys.path.append('/app/src')

try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    print('YOLOv8 model downloaded successfully')
except Exception as e:
    print(f'Model download failed: {e}')
    sys.exit(1)
"
    else
        log "YOLOv8 model already exists"
    fi
}

# Function to setup logging
setup_logging() {
    log "Setting up logging..."
    
    # Create log directories
    mkdir -p /app/logs /var/log/recipe-processing
    
    # Set appropriate permissions
    if [ "$ENVIRONMENT" = "development" ]; then
        chmod 755 /app/logs
    fi
    
    success "Logging setup completed"
}

# Function to run health check
health_check() {
    log "Running initial health check..."
    
    python -c "
import sys
sys.path.append('/app')
sys.path.append('/app/src')

try:
    from api.monitoring_logging import health_checker
    import asyncio
    
    async def check():
        results = await health_checker.check_all_components()
        for name, result in results.items():
            print(f'{name}: {result.status.value} - {result.message}')
        return all(result.status.value != 'unhealthy' for result in results.values())
    
    if asyncio.run(check()):
        print('Health check passed')
    else:
        print('Health check failed')
        sys.exit(1)
        
except Exception as e:
    print(f'Health check error: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        success "Health check passed"
    else
        warning "Health check failed, but continuing startup"
    fi
}

# Function to cleanup on exit
cleanup() {
    log "Shutting down gracefully..."
    
    # Kill background processes if any
    jobs -p | xargs -r kill
    
    log "Cleanup completed"
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Main startup sequence
main() {
    log "Starting Recipe Processing API..."
    log "Environment: $ENVIRONMENT"
    log "Python path: $PYTHONPATH"
    
    # Setup logging first
    setup_logging
    
    # Wait for external services
    wait_for_database
    wait_for_redis
    
    # Run database migrations
    run_migrations
    
    # Download required models
    download_models
    
    # Run health check
    health_check
    
    success "Initialization completed successfully"
    
    # Execute the main command
    log "Starting main application: $@"
    exec "$@"
}

# Help function
show_help() {
    echo "Recipe Processing API Docker Entrypoint"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  api         Start the API server (default)"
    echo "  worker      Start Celery worker"
    echo "  beat        Start Celery beat scheduler"
    echo "  flower      Start Flower monitoring"
    echo "  migrate     Run database migrations only"
    echo "  shell       Start Python shell"
    echo "  bash        Start bash shell"
    echo "  test        Run tests"
    echo "  help        Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  ENVIRONMENT         Environment (development/production)"
    echo "  DATABASE_URL        Database connection URL"
    echo "  REDIS_URL          Redis connection URL"
    echo "  LOG_LEVEL          Logging level (DEBUG/INFO/WARNING/ERROR)"
    echo ""
}

# Handle different commands
case "${1:-api}" in
    api)
        main python -m uvicorn src.api.recipe_processing_api:app \
            --host 0.0.0.0 --port 8000 \
            --workers ${WORKERS:-4} \
            --log-level ${LOG_LEVEL,,}
        ;;
        
    worker)
        # Skip model download for workers
        setup_logging
        wait_for_database
        wait_for_redis
        log "Starting Celery worker..."
        exec celery -A src.api.batch_processor worker \
            --loglevel=${LOG_LEVEL,,} \
            --concurrency=${WORKER_CONCURRENCY:-4}
        ;;
        
    beat)
        setup_logging
        wait_for_database
        wait_for_redis
        log "Starting Celery beat scheduler..."
        exec celery -A src.api.batch_processor beat \
            --loglevel=${LOG_LEVEL,,}
        ;;
        
    flower)
        setup_logging
        wait_for_redis
        log "Starting Flower monitoring..."
        exec celery -A src.api.batch_processor flower \
            --port=5555
        ;;
        
    migrate)
        setup_logging
        wait_for_database
        run_migrations
        ;;
        
    shell)
        setup_logging
        log "Starting Python shell..."
        exec python -c "
import sys
sys.path.append('/app')
sys.path.append('/app/src')
import IPython
IPython.start_ipython(argv=[])
"
        ;;
        
    bash)
        log "Starting bash shell..."
        exec /bin/bash
        ;;
        
    test)
        setup_logging
        log "Running tests..."
        exec python -m pytest tests/ -v
        ;;
        
    help|--help|-h)
        show_help
        exit 0
        ;;
        
    *)
        # Execute custom command
        main "$@"
        ;;
esac