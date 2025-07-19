#!/bin/bash
# Build and Deploy Script for Recipe Processing API
# Production deployment with health checks and rollback capability

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
IMAGE_NAME="${IMAGE_NAME:-recipe-processing-api}"
VERSION="${VERSION:-$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Service configuration
SERVICES=(
    "recipe-api"
    "celery-worker"
    "celery-beat"
    "flower"
)

# Health check configuration
HEALTH_CHECK_TIMEOUT=300  # 5 minutes
HEALTH_CHECK_INTERVAL=10  # 10 seconds

# Logging
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

# Help function
show_help() {
    echo "Recipe Processing API Build and Deploy Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build       Build Docker images"
    echo "  push        Push images to registry"
    echo "  deploy      Deploy to environment"
    echo "  rollback    Rollback to previous version"
    echo "  health      Check service health"
    echo "  logs        Show service logs"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  help        Show this help"
    echo ""
    echo "Options:"
    echo "  --environment ENV    Target environment (development/staging/production)"
    echo "  --version VERSION    Image version tag"
    echo "  --registry URL       Docker registry URL"
    echo "  --no-cache          Build without cache"
    echo "  --force             Force deployment without confirmation"
    echo ""
    echo "Environment Variables:"
    echo "  DOCKER_REGISTRY     Docker registry URL"
    echo "  IMAGE_NAME          Base image name"
    echo "  VERSION            Image version"
    echo "  ENVIRONMENT        Target environment"
    echo ""
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Function to build images
build_images() {
    local no_cache=""
    if [ "$1" = "--no-cache" ]; then
        no_cache="--no-cache"
        log "Building images without cache..."
    else
        log "Building images..."
    fi
    
    # Build production image
    docker build $no_cache \
        -t "${IMAGE_NAME}:${VERSION}" \
        -t "${IMAGE_NAME}:latest" \
        --target production \
        .
    
    if [ $? -eq 0 ]; then
        success "Production image built successfully"
    else
        error "Failed to build production image"
        exit 1
    fi
    
    # Build development image if in development environment
    if [ "$ENVIRONMENT" = "development" ]; then
        docker build $no_cache \
            -f Dockerfile.dev \
            -t "${IMAGE_NAME}:dev" \
            .
        
        if [ $? -eq 0 ]; then
            success "Development image built successfully"
        else
            warning "Failed to build development image"
        fi
    fi
    
    # Show built images
    log "Built images:"
    docker images | grep "$IMAGE_NAME"
}

# Function to push images to registry
push_images() {
    if [ "$DOCKER_REGISTRY" = "localhost:5000" ]; then
        warning "Using local registry, ensure it's running"
    fi
    
    log "Pushing images to $DOCKER_REGISTRY..."
    
    # Tag for registry
    docker tag "${IMAGE_NAME}:${VERSION}" "${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
    docker tag "${IMAGE_NAME}:latest" "${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"
    
    # Push images
    docker push "${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
    docker push "${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"
    
    if [ $? -eq 0 ]; then
        success "Images pushed successfully"
    else
        error "Failed to push images"
        exit 1
    fi
}

# Function to check service health
check_service_health() {
    local service_name=$1
    local timeout=${2:-60}
    local interval=${3:-5}
    
    log "Checking health of $service_name..."
    
    for i in $(seq 1 $((timeout / interval))); do
        if docker-compose ps | grep "$service_name" | grep -q "Up"; then
            # Check if service is actually healthy
            if docker-compose exec -T "$service_name" curl -f http://localhost:8000/health &> /dev/null; then
                success "$service_name is healthy"
                return 0
            fi
        fi
        
        if [ $i -eq $((timeout / interval)) ]; then
            error "$service_name failed health check after $timeout seconds"
            return 1
        fi
        
        sleep $interval
    done
}

# Function to wait for all services to be healthy
wait_for_services() {
    log "Waiting for all services to be healthy..."
    
    local all_healthy=true
    
    for service in "${SERVICES[@]}"; do
        if ! check_service_health "$service" 120 10; then
            all_healthy=false
            break
        fi
    done
    
    if [ "$all_healthy" = true ]; then
        success "All services are healthy"
        return 0
    else
        error "Some services failed health checks"
        return 1
    fi
}

# Function to deploy services
deploy_services() {
    local force=$1
    
    log "Deploying to $ENVIRONMENT environment..."
    
    # Choose compose file based on environment
    local compose_file="docker-compose.yml"
    if [ "$ENVIRONMENT" = "development" ]; then
        compose_file="docker-compose.dev.yml"
    elif [ "$ENVIRONMENT" = "staging" ]; then
        compose_file="docker-compose.staging.yml"
    fi
    
    if [ ! -f "$compose_file" ]; then
        error "Compose file $compose_file not found"
        exit 1
    fi
    
    # Confirmation for production
    if [ "$ENVIRONMENT" = "production" ] && [ "$force" != "--force" ]; then
        echo -n "Are you sure you want to deploy to PRODUCTION? (yes/no): "
        read -r confirmation
        if [ "$confirmation" != "yes" ]; then
            log "Deployment cancelled"
            exit 0
        fi
    fi
    
    # Create backup of current state
    log "Creating backup of current deployment..."
    docker-compose -f "$compose_file" config > "backup-${ENVIRONMENT}-$(date +%s).yml" 2>/dev/null || true
    
    # Update environment variables
    export IMAGE_TAG="$VERSION"
    export DOCKER_REGISTRY="$DOCKER_REGISTRY"
    
    # Pull latest images
    log "Pulling latest images..."
    docker-compose -f "$compose_file" pull
    
    # Stop existing services gracefully
    log "Stopping existing services..."
    docker-compose -f "$compose_file" down --timeout 30
    
    # Start new services
    log "Starting services..."
    docker-compose -f "$compose_file" up -d
    
    # Wait for services to be healthy
    if wait_for_services; then
        success "Deployment completed successfully"
        
        # Show running services
        log "Running services:"
        docker-compose -f "$compose_file" ps
    else
        error "Deployment failed - services are not healthy"
        log "Rolling back..."
        rollback_deployment
        exit 1
    fi
}

# Function to rollback deployment
rollback_deployment() {
    log "Rolling back deployment..."
    
    # Find the latest backup
    local backup_file=$(ls -t backup-${ENVIRONMENT}-*.yml 2>/dev/null | head -n1)
    
    if [ -n "$backup_file" ] && [ -f "$backup_file" ]; then
        log "Using backup: $backup_file"
        
        # Stop current services
        docker-compose down --timeout 30
        
        # Restore from backup
        docker-compose -f "$backup_file" up -d
        
        if wait_for_services; then
            success "Rollback completed successfully"
        else
            error "Rollback failed"
            exit 1
        fi
    else
        error "No backup found for rollback"
        exit 1
    fi
}

# Function to show service logs
show_logs() {
    local service=${1:-}
    local follow=${2:-}
    
    if [ -n "$service" ]; then
        log "Showing logs for $service..."
        if [ "$follow" = "-f" ]; then
            docker-compose logs -f "$service"
        else
            docker-compose logs --tail=100 "$service"
        fi
    else
        log "Showing logs for all services..."
        if [ "$follow" = "-f" ]; then
            docker-compose logs -f
        else
            docker-compose logs --tail=50
        fi
    fi
}

# Function to stop services
stop_services() {
    log "Stopping all services..."
    docker-compose down --timeout 30
    success "All services stopped"
}

# Function to restart services
restart_services() {
    log "Restarting all services..."
    docker-compose restart
    
    if wait_for_services; then
        success "All services restarted successfully"
    else
        error "Some services failed to restart properly"
        exit 1
    fi
}

# Function to clean up old images
cleanup_images() {
    log "Cleaning up old images..."
    
    # Remove dangling images
    docker image prune -f
    
    # Remove old versions (keep last 5)
    docker images "${IMAGE_NAME}" --format "table {{.Tag}}\t{{.ID}}" | \
        grep -v latest | \
        tail -n +6 | \
        awk '{print $2}' | \
        xargs -r docker rmi
    
    success "Image cleanup completed"
}

# Main function
main() {
    local command="${1:-help}"
    shift || true
    
    # Parse options
    local no_cache=""
    local force=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            --registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            --no-cache)
                no_cache="--no-cache"
                shift
                ;;
            --force)
                force="--force"
                shift
                ;;
            *)
                warning "Unknown option: $1"
                shift
                ;;
        esac
    done
    
    # Check prerequisites
    check_prerequisites
    
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    log "Registry: $DOCKER_REGISTRY"
    
    # Execute command
    case $command in
        build)
            build_images $no_cache
            ;;
        push)
            push_images
            ;;
        deploy)
            build_images $no_cache
            deploy_services $force
            ;;
        rollback)
            rollback_deployment
            ;;
        health)
            wait_for_services
            ;;
        logs)
            show_logs "$1" "$2"
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        cleanup)
            cleanup_images
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"