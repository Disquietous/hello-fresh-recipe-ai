#!/bin/bash
# Performance Testing and Optimization Script
# Comprehensive performance testing suite for Recipe Processing API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/benchmarks/results"
TEST_IMAGES_DIR="${PROJECT_ROOT}/data/test/images"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"

# Performance test configuration
DEFAULT_CONCURRENCY=(1 2 4 8 16)
DEFAULT_DURATIONS=(30 60 120)
DEFAULT_TEST_TYPES=("api" "pipeline" "cache" "memory" "cpu")

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
    echo "Recipe Processing API Performance Testing Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  benchmark     Run comprehensive performance benchmarks"
    echo "  load-test     Run load testing with specified parameters"
    echo "  api-test      Test API endpoints performance"
    echo "  stress-test   Run stress testing with high load"
    echo "  profile       Run CPU and memory profiling"
    echo "  optimize      Generate optimization recommendations"
    echo "  compare       Compare performance between versions"
    echo "  report        Generate performance report from existing results"
    echo "  clean         Clean up old test results"
    echo "  help          Show this help"
    echo ""
    echo "Options:"
    echo "  --concurrency NUMS    Concurrent request levels (default: 1,2,4,8,16)"
    echo "  --duration SECS       Test duration in seconds (default: 30,60,120)"
    echo "  --output DIR          Output directory for results (default: benchmarks/results)"
    echo "  --test-images DIR     Test images directory (default: data/test/images)"
    echo "  --environment ENV     Test environment (development/staging/production)"
    echo "  --api-url URL         API base URL for testing (default: http://localhost:8000)"
    echo "  --skip-setup          Skip test environment setup"
    echo "  --verbose             Enable verbose logging"
    echo "  --format FORMAT       Output format (json/csv/html) (default: json)"
    echo ""
}

# Function to setup test environment
setup_test_environment() {
    log "Setting up test environment..."
    
    # Create necessary directories
    mkdir -p "$RESULTS_DIR" "$TEST_IMAGES_DIR"
    
    # Check if Docker services are running
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        log "Checking Docker services..."
        if ! docker-compose ps | grep -q "Up"; then
            warning "Starting Docker services for testing..."
            docker-compose up -d
            sleep 30  # Wait for services to be ready
        fi
    fi
    
    # Generate test images if needed
    if [ ! "$(ls -A "$TEST_IMAGES_DIR" 2>/dev/null)" ]; then
        log "Generating test images..."
        python3 -c "
import sys, os
sys.path.append('$PROJECT_ROOT')
from src.performance_benchmarking import PerformanceBenchmark, BenchmarkConfig
config = BenchmarkConfig(test_images_dir='$TEST_IMAGES_DIR', output_dir='$RESULTS_DIR')
benchmark = PerformanceBenchmark(config)
benchmark.generate_test_images()
print(f'Test images generated in $TEST_IMAGES_DIR')
"
    fi
    
    success "Test environment setup completed"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Python and required packages
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if required Python packages are available
    python3 -c "
import sys
required_packages = ['numpy', 'matplotlib', 'seaborn', 'psutil', 'PIL']
missing = []
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing.append(package)
if missing:
    print(f'Missing required packages: {missing}')
    sys.exit(1)
" || {
        error "Required Python packages are missing. Run: pip install -r requirements.txt"
        exit 1
    }
    
    # Check available system resources
    local available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$available_memory" -lt 1000 ]; then
        warning "Low available memory ($available_memory MB). Consider reducing test concurrency."
    fi
    
    # Check disk space
    local available_disk=$(df "$RESULTS_DIR" | awk 'NR==2{printf "%.0f", $4/1024}')
    if [ "$available_disk" -lt 1000 ]; then
        warning "Low available disk space ($available_disk MB) in results directory."
    fi
    
    success "Prerequisites check passed"
}

# Function to run comprehensive benchmarks
run_benchmark() {
    local concurrency_levels="$1"
    local test_durations="$2"
    local skip_setup="$3"
    local verbose="$4"
    
    log "Running comprehensive performance benchmarks..."
    
    if [ "$skip_setup" != "true" ]; then
        setup_test_environment
    fi
    
    # Prepare Python arguments
    local python_args="--output-dir '$RESULTS_DIR' --test-images '$TEST_IMAGES_DIR'"
    
    if [ -n "$concurrency_levels" ]; then
        python_args="$python_args --concurrency $(echo $concurrency_levels | tr ',' ' ')"
    fi
    
    if [ -n "$test_durations" ]; then
        python_args="$python_args --duration $(echo $test_durations | tr ',' ' ')"
    fi
    
    if [ "$verbose" = "true" ]; then
        python_args="$python_args --verbose"
    fi
    
    # Run the benchmark
    log "Starting Python benchmark script..."
    cd "$PROJECT_ROOT"
    
    if python3 src/performance_benchmarking.py $python_args; then
        success "Benchmark completed successfully"
        
        # Generate optimization recommendations
        python3 src/performance_benchmarking.py --optimize --output-dir "$RESULTS_DIR"
        
        # Show summary
        if [ -f "$RESULTS_DIR/performance_summary.json" ]; then
            log "Performance Summary:"
            python3 -c "
import json
with open('$RESULTS_DIR/performance_summary.json', 'r') as f:
    data = json.load(f)
for operation, metrics in data.items():
    print(f'{operation}: {metrics[\"avg_duration\"]:.3f}s avg, {metrics[\"success_rate\"]*100:.1f}% success')
"
        fi
    else
        error "Benchmark failed"
        exit 1
    fi
}

# Function to run API load testing
run_api_load_test() {
    local api_url="$1"
    local concurrency="$2"
    local duration="$3"
    local test_image="$4"
    
    log "Running API load test: $concurrency concurrent for ${duration}s"
    
    # Check if API is accessible
    if ! curl -s --head "$api_url/health" > /dev/null; then
        error "API at $api_url is not accessible"
        return 1
    fi
    
    # Create test script for API load testing
    local load_test_script="$RESULTS_DIR/api_load_test.py"
    cat > "$load_test_script" << 'EOF'
import asyncio
import aiohttp
import time
import json
import sys
from pathlib import Path

async def send_request(session, url, image_path, semaphore):
    async with semaphore:
        try:
            with open(image_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename='test.jpg', content_type='image/jpeg')
                
                start_time = time.time()
                async with session.post(f"{url}/process", data=data) as response:
                    duration = time.time() - start_time
                    return {
                        'status': response.status,
                        'duration': duration,
                        'success': response.status == 200
                    }
        except Exception as e:
            return {
                'status': 0,
                'duration': 0,
                'success': False,
                'error': str(e)
            }

async def run_load_test(api_url, concurrency, duration, image_path):
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < duration:
            task = send_request(session, api_url, image_path, semaphore)
            result = await task
            results.append(result)
            
            if len(results) % 10 == 0:
                print(f"Completed {len(results)} requests...")
    
    # Calculate statistics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    if successful:
        avg_duration = sum(r['duration'] for r in successful) / len(successful)
        min_duration = min(r['duration'] for r in successful)
        max_duration = max(r['duration'] for r in successful)
    else:
        avg_duration = min_duration = max_duration = 0
    
    stats = {
        'total_requests': len(results),
        'successful_requests': len(successful),
        'failed_requests': len(failed),
        'success_rate': len(successful) / len(results) if results else 0,
        'requests_per_second': len(results) / duration,
        'avg_response_time': avg_duration,
        'min_response_time': min_duration,
        'max_response_time': max_duration
    }
    
    return stats

if __name__ == "__main__":
    api_url = sys.argv[1]
    concurrency = int(sys.argv[2])
    duration = int(sys.argv[3])
    image_path = sys.argv[4]
    
    stats = asyncio.run(run_load_test(api_url, concurrency, duration, image_path))
    print(json.dumps(stats, indent=2))
EOF
    
    # Find a test image
    if [ -z "$test_image" ] || [ ! -f "$test_image" ]; then
        test_image=$(find "$TEST_IMAGES_DIR" -name "*.jpg" -o -name "*.png" | head -1)
        if [ -z "$test_image" ]; then
            error "No test images found in $TEST_IMAGES_DIR"
            return 1
        fi
    fi
    
    # Run the load test
    log "Using test image: $test_image"
    local output_file="$RESULTS_DIR/api_load_test_${concurrency}c_${duration}s.json"
    
    if python3 "$load_test_script" "$api_url" "$concurrency" "$duration" "$test_image" > "$output_file"; then
        success "API load test completed. Results saved to $output_file"
        
        # Show summary
        python3 -c "
import json
with open('$output_file', 'r') as f:
    stats = json.load(f)
print(f'Total requests: {stats[\"total_requests\"]}')
print(f'Success rate: {stats[\"success_rate\"]*100:.1f}%')
print(f'Requests/sec: {stats[\"requests_per_second\"]:.2f}')
print(f'Avg response time: {stats[\"avg_response_time\"]*1000:.0f}ms')
"
    else
        error "API load test failed"
        return 1
    fi
}

# Function to run stress testing
run_stress_test() {
    local api_url="$1"
    local max_concurrency="$2"
    local step_duration="$3"
    
    log "Running stress test up to $max_concurrency concurrent requests"
    
    local stress_results="$RESULTS_DIR/stress_test_results.json"
    echo "[]" > "$stress_results"
    
    # Find test image
    local test_image=$(find "$TEST_IMAGES_DIR" -name "*.jpg" -o -name "*.png" | head -1)
    if [ -z "$test_image" ]; then
        error "No test images found for stress testing"
        return 1
    fi
    
    # Run stress test with increasing concurrency
    for concurrency in $(seq 1 $max_concurrency); do
        log "Stress testing with $concurrency concurrent requests..."
        
        local temp_result="$RESULTS_DIR/stress_temp_${concurrency}.json"
        
        if run_api_load_test "$api_url" "$concurrency" "$step_duration" "$test_image"; then
            # Merge results
            python3 -c "
import json
with open('$stress_results', 'r') as f:
    results = json.load(f)
with open('$RESULTS_DIR/api_load_test_${concurrency}c_${step_duration}s.json', 'r') as f:
    new_result = json.load(f)
new_result['concurrency'] = $concurrency
results.append(new_result)
with open('$stress_results', 'w') as f:
    json.dump(results, f, indent=2)
"
            
            # Check if we've hit limits
            local success_rate=$(python3 -c "
import json
with open('$RESULTS_DIR/api_load_test_${concurrency}c_${step_duration}s.json', 'r') as f:
    data = json.load(f)
print(data['success_rate'])
")
            
            if [ "$(echo "$success_rate < 0.9" | bc -l)" -eq 1 ]; then
                warning "Success rate dropped below 90% at $concurrency concurrent requests"
                break
            fi
        else
            warning "Stress test failed at $concurrency concurrent requests"
            break
        fi
        
        # Cleanup temporary files
        rm -f "$temp_result"
    done
    
    success "Stress test completed. Results in $stress_results"
}

# Function to generate performance report
generate_report() {
    local output_format="$1"
    local report_file="$RESULTS_DIR/performance_report"
    
    log "Generating performance report in $output_format format..."
    
    # Create comprehensive report script
    local report_script="$RESULTS_DIR/generate_report.py"
    cat > "$report_script" << 'EOF'
import json
import sys
from pathlib import Path
from datetime import datetime

def generate_html_report(results_dir, output_file):
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Recipe Processing Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ background: #e9f7ef; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Recipe Processing Performance Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
    
    # Add summary if available
    summary_file = Path(results_dir) / "performance_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        
        html_content += '<div class="section"><h2>Performance Summary</h2><table>'
        html_content += '<tr><th>Operation</th><th>Avg Duration (s)</th><th>Success Rate (%)</th><th>Avg Throughput (ops/s)</th></tr>'
        
        for operation, metrics in summary.items():
            html_content += f'<tr><td>{operation}</td><td>{metrics["avg_duration"]:.3f}</td><td>{metrics["success_rate"]*100:.1f}</td><td>{metrics["avg_throughput"]:.2f}</td></tr>'
        
        html_content += '</table></div>'
    
    html_content += '</body></html>'
    
    with open(output_file, 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    results_dir = sys.argv[1]
    output_format = sys.argv[2]
    output_file = sys.argv[3]
    
    if output_format == "html":
        generate_html_report(results_dir, output_file)
    else:
        print(f"Format {output_format} not yet implemented")
EOF
    
    case $output_format in
        "html")
            python3 "$report_script" "$RESULTS_DIR" "html" "${report_file}.html"
            success "HTML report generated: ${report_file}.html"
            ;;
        "json")
            if [ -f "$RESULTS_DIR/performance_summary.json" ]; then
                cp "$RESULTS_DIR/performance_summary.json" "${report_file}.json"
                success "JSON report generated: ${report_file}.json"
            else
                error "No performance summary found"
                return 1
            fi
            ;;
        *)
            error "Unsupported report format: $output_format"
            return 1
            ;;
    esac
}

# Function to clean up old results
clean_results() {
    log "Cleaning up old test results..."
    
    if [ -d "$RESULTS_DIR" ]; then
        # Keep results from last 7 days
        find "$RESULTS_DIR" -name "*.json" -mtime +7 -delete
        find "$RESULTS_DIR" -name "*.csv" -mtime +7 -delete
        find "$RESULTS_DIR" -name "*.png" -mtime +7 -delete
        find "$RESULTS_DIR" -name "*.prof" -mtime +7 -delete
        
        success "Old results cleaned up"
    else
        warning "Results directory does not exist"
    fi
}

# Main function
main() {
    local command="${1:-help}"
    shift || true
    
    # Parse options
    local concurrency_levels=""
    local test_durations=""
    local output_dir="$RESULTS_DIR"
    local test_images_dir="$TEST_IMAGES_DIR"
    local api_url="http://localhost:8000"
    local environment="development"
    local skip_setup="false"
    local verbose="false"
    local output_format="json"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --concurrency)
                concurrency_levels="$2"
                shift 2
                ;;
            --duration)
                test_durations="$2"
                shift 2
                ;;
            --output)
                output_dir="$2"
                RESULTS_DIR="$output_dir"
                shift 2
                ;;
            --test-images)
                test_images_dir="$2"
                TEST_IMAGES_DIR="$test_images_dir"
                shift 2
                ;;
            --api-url)
                api_url="$2"
                shift 2
                ;;
            --environment)
                environment="$2"
                shift 2
                ;;
            --skip-setup)
                skip_setup="true"
                shift
                ;;
            --verbose)
                verbose="true"
                shift
                ;;
            --format)
                output_format="$2"
                shift 2
                ;;
            *)
                warning "Unknown option: $1"
                shift
                ;;
        esac
    done
    
    # Check prerequisites
    check_prerequisites
    
    log "Configuration:"
    log "  Results directory: $RESULTS_DIR"
    log "  Test images: $TEST_IMAGES_DIR"
    log "  API URL: $api_url"
    log "  Environment: $environment"
    
    # Execute command
    case $command in
        benchmark)
            run_benchmark "$concurrency_levels" "$test_durations" "$skip_setup" "$verbose"
            ;;
        load-test)
            setup_test_environment
            run_api_load_test "$api_url" "${concurrency_levels:-4}" "${test_durations:-60}" ""
            ;;
        api-test)
            setup_test_environment
            for concurrency in ${concurrency_levels:-1 2 4 8}; do
                run_api_load_test "$api_url" "$concurrency" "30" ""
            done
            ;;
        stress-test)
            setup_test_environment
            run_stress_test "$api_url" "${concurrency_levels:-20}" "30"
            ;;
        profile)
            run_benchmark "$concurrency_levels" "$test_durations" "$skip_setup" "$verbose"
            ;;
        optimize)
            python3 src/performance_benchmarking.py --optimize --output-dir "$RESULTS_DIR"
            ;;
        compare)
            # TODO: Implement version comparison
            warning "Version comparison not yet implemented"
            ;;
        report)
            generate_report "$output_format"
            ;;
        clean)
            clean_results
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